use super::{END_OF_STRING, START_OF_STRING};

pub struct PreProcessor {
    sent: String,
}

impl PreProcessor {
    pub fn new(sentence: String) -> Self {
        Self { sent: sentence }
    }
}

impl PreProcessor {
    pub fn lowercase(mut self) -> Self {
        self.sent.make_ascii_lowercase();
        self
    }
    pub fn add_start_end_tokens(mut self) -> Self {
        self.sent = format!("{} {} {}", START_OF_STRING, self.sent, END_OF_STRING);
        self
    }
    pub fn done(self) -> String {
        self.sent.to_owned()
    }
}
