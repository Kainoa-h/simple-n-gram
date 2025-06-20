use super::PreProcessor;
use super::{END_OF_STRING, START_OF_STRING};

pub struct StartEndTokensPreProcessor {
    next: Option<Box<dyn PreProcessor>>,
}

impl Default for StartEndTokensPreProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StartEndTokensPreProcessor {
    pub fn new() -> Self {
        Self { next: None }
    }
}

impl PreProcessor for StartEndTokensPreProcessor {
    fn process(&self, sentence: String) -> Result<String, String> {
        self.pass(format!(
            "{} {} {}",
            START_OF_STRING, sentence, END_OF_STRING
        ))
    }

    fn set_next(&mut self, next: Box<dyn PreProcessor>) {
        self.next = Some(next);
    }

    fn get_next(&self) -> &Option<Box<dyn PreProcessor>> {
        &self.next
    }
}
