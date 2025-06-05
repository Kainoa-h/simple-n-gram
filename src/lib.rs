/*
Features:

- <unk> tag for out of vocab words
- Enum for choosing
*/
use rand::seq::SliceRandom;
use std::fs::{self};
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    thread::AccessError,
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

pub struct LowerCasePreProcessor {
    next: Option<Box<dyn PreProcessor>>,
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

pub struct StartEndTokensPreProcessor {
    next: Option<Box<dyn PreProcessor>>,
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

pub trait Model {
    type ModelError;

    fn build_n_gram(
        &mut self, pre_processor_chain: Box<dyn PreProcessor>, corpus: Vec<&str>,
    ) -> Result<(), String>;

    fn predict(&self, max_tokens: u32);

    fn save(&self) -> Result<String, <Self as Model>::ModelError>;

    fn load(&mut self) -> Result<(), String>;
}

pub struct LidstoneModel {
    n_size: usize,
    alpha: f32,
    top_k: f32,
    temperature: f32,
    lower_case: bool,
    vocabulary: HashSet<String>, //TODO: not sure what this is for
    n_gram_map: HashMap<String, Vec<(String, u32, f64)>>,
    start_tokens: Vec<String>,
}

impl LidstoneModel {
    pub fn new(config: Config) -> Self {
        Self {
            n_size: config.n_size,
            alpha: config.lidstone_alpha,
            top_k: config.top_k,
            temperature: config.temperature,
            lower_case: config.lower_case,
            vocabulary: HashSet::new(),
            n_gram_map: HashMap::new(),
            start_tokens: Vec::new(),
        }
    }
}

type PredictFn = fn(&mut LidstoneModel);

type PredictFn = fn(&mut LidstoneModel);

type PredictFn = fn(&mut LidstoneModel);

impl Model for LidstoneModel {
    type ModelError = String;

    fn build_n_gram(
        &mut self, pre_processor_chain: Box<dyn PreProcessor>, corpus: Vec<&str>,
    ) -> Result<(), String> {
        let mut n_gram_map_builder: HashMap<String, HashMap<String, u32>> = HashMap::new();

        for sentence in corpus {
            let tokenized_sent = pre_processor_chain
                .process(sentence.to_owned())?
                .split_whitespace()
                .map(|x| x.to_owned())
                .collect::<Vec<String>>();

            if tokenized_sent.len() < self.n_size {
                return Err(format!(
                    "Length of sentence is shorter than ngram size of {}:\n\"{}\"",
                    self.n_size, sentence
                ));
            }

            self.vocabulary.extend(tokenized_sent.iter().cloned());
            self.start_tokens.push(
                tokenized_sent[0..self.n_size - 1]
                    .iter()
                    .map(|w| w.to_owned())
                    .collect(),
            );

            for i in 0..=tokenized_sent.len() - self.n_size {
                let ctx_tokens: String = tokenized_sent[i..i + self.n_size - 1]
                    .iter()
                    .map(|w| w.to_owned())
                    .collect();
                let target_token: String = tokenized_sent[i + self.n_size - 1].to_owned();

                n_gram_map_builder
                    .entry(ctx_tokens.clone())
                    .and_modify(|x| {
                        x.entry(target_token.clone())
                            .and_modify(|y| *y += 1)
                            .or_insert(1);
                    })
                    .or_insert_with(|| HashMap::from([(target_token.clone(), 1)]));
            }
        }

        for (context_token, candidates) in n_gram_map_builder.iter() {
            let mut candidate_tuples: Vec<(String, u32, f64)> = Vec::new();
            let mut total: u32 = 0;
            for (candidate_token, count) in candidates {
                total += count;
                candidate_tuples.push((candidate_token.to_owned(), *count, 0.0));
            }
            for tup in candidate_tuples.iter_mut() {
                tup.2 = tup.1 as f64 / total as f64;
            }
            candidate_tuples.sort_by(|a, b| b.1.cmp(&a.1));
            self.n_gram_map
                .insert(context_token.to_owned(), candidate_tuples);
        }

        Ok(())
    }

    fn predict(&self, max_tokens: u32) {
        let mut count: u32 = 0;
        let output: String = String::new();
        loop {
            if max_tokens != 0 && max_tokens == count {
                return;
            };
            //TODO:randonly find a starting string

            if max_tokens != 0 {
                count += 1;
            }
        }
    }

    fn save(&self) -> Result<String, <LidstoneModel as Model>::ModelError> {
        let simplified: HashMap<String, Vec<(String, u32)>> = self
            .n_gram_map
            .iter()
            .map(|(ctx, vec)| {
                let candidates = vec.iter().map(|(s, u, _)| (s.clone(), *u)).collect();
                (ctx.clone(), candidates)
            })
            .collect();

        let model = (self.start_tokens.clone(), simplified);

        let model_json = serde_json::to_string(&model).expect("lol");
        match fs::write("model.json", &model_json) {
            Ok(_) => Ok(model_json),
            Err(err) => Err(format!("File write error: {}", err)),
        }
    }

    fn load(&mut self) -> Result<(), String> {
        todo!()
    }
}
