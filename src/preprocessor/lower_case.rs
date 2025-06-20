use super::PreProcessor;

pub struct LowerCasePreProcessor {
    next: Option<Box<dyn PreProcessor>>,
}

impl Default for LowerCasePreProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LowerCasePreProcessor {
    pub fn new() -> Self {
        Self { next: None }
    }
}

impl PreProcessor for LowerCasePreProcessor {
    fn process(&self, sentence: String) -> Result<String, String> {
        self.pass(sentence.to_lowercase())
    }

    fn set_next(&mut self, next: Box<dyn PreProcessor>) {
        self.next = Some(next);
    }

    fn get_next(&self) -> &Option<Box<dyn PreProcessor>> {
        &self.next
    }
}
