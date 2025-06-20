pub trait PreProcessor {
    fn process(&self, sentence: String) -> Result<String, String>;
    fn set_next(&mut self, next: Box<dyn PreProcessor>);
    fn get_next(&self) -> &Option<Box<dyn PreProcessor>>;

    fn pass(&self, sentence: String) -> Result<String, String> {
        if let Some(next) = self.get_next() {
            next.process(sentence)
        } else {
            Ok(sentence)
        }
    }
}
