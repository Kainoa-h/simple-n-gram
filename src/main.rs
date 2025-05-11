/*
Features:

- <unk> tag for out of vocab words
- Enum for choosing
*/
use rand::seq::SliceRandom;
use std::{
  cmp::max,
  collections::{HashMap, HashSet},
};

const START_OF_STRING: &str = "<s>";
const END_OF_STRING: &str = "<s/>";

enum SmoothingType {
  Lidstone,
}

pub struct Config {
  n_size: usize,
  smoothing_type: SmoothingType,
  lidstone_alpha: f32,
  top_k: f32,
  temperature: f32,
  lower_case: bool,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      n_size: 2,
      smoothing_type: SmoothingType::Lidstone,
      lidstone_alpha: 1.0,
      top_k: 0.8,
      temperature: 1.0,
      lower_case: true,
    }
  }
}

trait Model {
  fn train(&mut self, corpus: Vec<String>) -> Result<(), String>;

  fn predict(&mut self);

  fn get_lower_case_config(&self) -> bool;

  fn tokenize(&mut self, sentence: String) -> Vec<String> {
    let sent = format!("{} {} {}", START_OF_STRING, sentence, END_OF_STRING);
    if self.get_lower_case_config() {
      sent.to_lowercase()
    } else {
      sent
    }
    .split_whitespace()
    .map(|t| t.to_string())
    .collect()
  }

  fn save(&self) -> Result<(), String>;

  fn load(&mut self) -> Result<(), String>;
}

trait PreProcessor {
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

struct LowerCasePreProcessor {
  next: Option<Box<dyn PreProcessor>>,
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

struct StartEndTokensPreProcessor {
  next : Option<Box<dyn PreProcessor>>,
}
impl PreProcessor for StartEndTokensPreProcessor {
  fn process(&self, sentence: String) -> Result<String, String> {
    self.pass(format!("{} {} {}", START_OF_STRING, sentence, END_OF_STRING))
  }

  fn set_next(&mut self, next: Box<dyn PreProcessor>) {
    self.next = Some(next);
  }

  fn get_next(&self) -> &Option<Box<dyn PreProcessor>> {
    &self.next
  }
}

pub struct LidstoneModel {
  n_size: usize,
  alpha: f32,
  top_k: f32,
  temperature: f32,
  lower_case: bool,
  vocabulary: HashSet<String>,
  n_gram_map: HashMap<String, HashMap<String, u32>>,
}

impl LidstoneModel {
  fn new(config: Config) -> Self {
    Self {
      n_size: config.n_size,
      alpha: config.lidstone_alpha,
      top_k: config.top_k,
      temperature: config.temperature,
      lower_case: config.lower_case,
      vocabulary: HashSet::new(),
      n_gram_map: HashMap::new(),
    }
  }
}

impl Model for LidstoneModel {
  fn train(&mut self, corpus: Vec<String>) -> Result<(), String> {
    for sentence in corpus {
      let tokenized_sent = self.tokenize(sentence.clone());
      if tokenized_sent.len() < self.n_size {
        return Err(format!(
          "Length of sentence is shorter than ngram sizeof {}:\n\"{}\"",
          self.n_size, sentence
        ));
      }
      for i in 0..tokenized_sent.len() {
        self.vocabulary.insert(tokenized_sent[i].clone());
      }

      for i in 0..=tokenized_sent.len() - self.n_size {
        let context_word: String = tokenized_sent[i..i + self.n_size - 1]
          .iter()
          .map(|w| w.to_string())
          .collect();
        let target_word: String = tokenized_sent[i + self.n_size].to_owned();

        self
          .n_gram_map
          .entry(context_word.clone())
          .and_modify(|x| {
            x.entry(target_word.clone())
              .and_modify(|y| *y += 1)
              .or_insert(1);
          })
          .or_insert_with(|| HashMap::from([(target_word.clone(), 1)]));
      }
    }
    Ok(())
  }

  fn predict(&mut self) {
    todo!()
  }

  fn get_lower_case_config(&self) -> bool {
    self.lower_case.to_owned()
  }

  fn save(&self) -> Result<(), String> {
    todo!()
  }

  fn load(&mut self) -> Result<(), String> {
    todo!()
  }
}

fn main() {
  println!("Hello, world!");
}
