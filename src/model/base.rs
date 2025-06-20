use rand::rngs::StdRng;

pub trait Model: Sized {
    type ModelError;

    fn build_n_gram(
        &mut self, pre_processor_chain: Box<dyn crate::preprocessor::PreProcessor>,
        corpus: Vec<&str>,
    ) -> Result<(), String>;

    fn generate(&self, max_tokens: u32, seed: u64) -> String;

    fn predict_next_token(&self, context: &str, rng: &mut StdRng) -> &str;

    fn save(&self) -> Result<String, Self::ModelError>;

    fn load(path: &str) -> Result<Self, String>;
}
