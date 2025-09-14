// axum_ort_single_file.rs (fixed)
// Axum + ort (Option B, download-binaries). POST /score -> { risk: f32 }
// - No `inputs!` macro (not present in ort 1.16).
// - Uses `Value::from_array(session.allocator(), &CowArray)`.
// Cargo.toml deps:
//   axum = "0.7"
//   tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
//   serde = { version = "1", features = ["derive"] }
//   serde_json = "1"
//   anyhow = "1"
//   ndarray = "0.15"
//   ort = { version = "1.16", features = ["download-binaries"] }
//
// Run:
//   MODEL_PATH=./tiny_mlp_evaluator.onnx cargo run --release
//
// Test:
//   curl -X POST http://localhost:8080/score \
//        -H "content-type: application/json" \
//        -d '{"features":[0.9,0.75,0.12,0.40,1.23,0.98]}'

use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr, sync::Arc};
use anyhow::Result;
use tokio::net::TcpListener;

use ndarray::{Array2, CowArray, IxDyn};

// ort (ONNX Runtime)
use ort::environment::Environment;
use ort::session::{Session, SessionBuilder};
use ort::{GraphOptimizationLevel, LoggingLevel};
use ort::tensor::OrtOwnedTensor;
use ort::value::Value;

#[derive(Clone)]
struct AppState {
    session: Arc<Session>,
}

#[derive(Deserialize)]
struct ScoreRequest {
    // For your evaluator this should be length 6 (unstandardized features)
    features: Vec<f32>,
}

#[derive(Serialize)]
struct ScoreResponse {
    // Probability in [0,1]
    risk: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Config
    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "tiny_mlp_evaluator.onnx".to_string());
    let addr: SocketAddr = env::var("ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()
        .expect("ADDR like 0.0.0.0:8080");

    // ort environment (Arc) + session
    let env = Arc::new(
        Environment::builder()
            .with_name("evaluator")
            .with_log_level(LoggingLevel::Warning)
            .build()?
    );

    let session = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_model_from_file(&model_path)?;

    let state = AppState { session: Arc::new(session) };

    // Axum 0.7 server style
    let app = Router::new().route("/score", post(score)).with_state(state);
    let listener = TcpListener::bind(addr).await?;
    println!("Serving on http://{} (MODEL_PATH={})", addr, model_path);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn score(
    State(state): State<AppState>,
    Json(req): Json<ScoreRequest>,
) -> Result<Json<ScoreResponse>, (axum::http::StatusCode, String)> {
    if req.features.is_empty() {
        return Err((axum::http::StatusCode::BAD_REQUEST, "features cannot be empty".into()));
    }

    // Build [batch=1, feat_dim] ndarray. ONNX expects UNSTANDARDIZED features.
    let feat_dim = req.features.len();
    let input: Array2<f32> = Array2::from_shape_vec((1, feat_dim), req.features)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("bad features shape: {e}")))?;

    // Convert Array2<f32> -> dynamic-dim CowArray -> ort::Value
    let dyn_arr = input.into_dyn();                 // Array<f32, IxDyn>
    let cow: CowArray<f32, IxDyn> = CowArray::from(dyn_arr);
    let val: Value = Value::from_array(state.session.allocator(), &cow)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("to-onnx value error: {e}")))?;

    // Inference
    let outputs = state.session
        .run(vec![val])
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("inference error: {e}")))?;

    // Extract first scalar from first output
    let tensor: OrtOwnedTensor<'_, f32, IxDyn> = outputs[0]
        .try_extract()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("bad output type: {e}")))?;
    let risk = *tensor.view().as_slice().and_then(|s| s.first()).unwrap_or(&0.0);

    Ok(Json(ScoreResponse { risk }))
}
