/*
Features:

- <unk> tag for out of vocab words
- Enum for choosing
*/
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::{self, File};
use std::io::BufReader;
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
};

const START_OF_STRING: &str = "<s>";
const END_OF_STRING: &str = "<s/>";

pub enum SmoothingType {
    Lidstone,
}

pub struct Config {
    pub n_size: usize,
    pub smoothing_type: SmoothingType,
    pub lidstone_alpha: f64,
    pub top_k: f64,
    pub temperature: f64,
    pub lower_case: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_size: 2,
            smoothing_type: SmoothingType::Lidstone,
            lidstone_alpha: 1.0,
            top_k: 1.0,
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

pub trait Model {
    type ModelError;

    fn build_n_gram(
        &mut self, pre_processor_chain: Box<dyn PreProcessor>, corpus: Vec<&str>,
    ) -> Result<(), String>;

    fn generate(&self, max_tokens: u32) -> String;

    fn predict_next_token(&self, context: &str, rng: &mut StdRng) -> &str;

    fn save(&self) -> Result<String, <Self as Model>::ModelError>;

    fn load(&mut self, path: &str) -> Result<(), String>;
}

pub struct LidstoneModel {
    n_size: usize,
    alpha: f64,
    top_k: f64,
    temperature: f64,
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

impl Model for LidstoneModel {
    type ModelError = String;

    fn build_n_gram(
        &mut self, pre_processor_chain: Box<dyn PreProcessor>, corpus: Vec<&str>,
    ) -> Result<(), String> {
        let mut n_gram_map_builder: HashMap<String, HashMap<String, u32>> = HashMap::new();

        for (idx, &sentence) in corpus.iter().enumerate() {
            if sentence.is_empty() {
                continue;
            };
            let tokenized_sent = pre_processor_chain
                .process(sentence.to_owned())?
                .split_whitespace()
                .map(|x| x.to_owned())
                .collect::<Vec<String>>();

            if tokenized_sent.len() < self.n_size {
                return Err(format!(
                    "Length of sentence at line {} is shorter than ngram size of {}:\n\"{}\"",
                    idx, self.n_size, sentence
                ));
            }

            self.vocabulary.extend(tokenized_sent.iter().cloned());
            self.start_tokens.push(
                tokenized_sent[0..self.n_size - 1]
                    .iter()
                    .map(|w| w.as_str())
                    .collect::<Vec<&str>>()
                    .join(" "),
            );

            for i in 0..=tokenized_sent.len() - self.n_size {
                let ctx_tokens: String = tokenized_sent[i..i + self.n_size - 1]
                    .iter()
                    .map(|w| w.as_str())
                    .collect::<Vec<&str>>()
                    .join(" ");
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
            candidate_tuples.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            self.n_gram_map
                .insert(context_token.to_owned(), candidate_tuples);
        }

        Ok(())
    }

    fn generate(&self, max_tokens: u32) -> String {
        let mut rng = StdRng::seed_from_u64(54);
        let mut count: u32 = 0;
        let mut output: Vec<&str> = Vec::new();
        let start: &str = &self.start_tokens[rng.random_range(0..self.start_tokens.len())];
        output.extend(start.split(' '));
        let next: &str = self.predict_next_token(start, &mut rng);
        output.push(next);
        loop {
            if max_tokens != 0 && max_tokens == count {
                return output.join(" ");
            };
            let sidx = output.len() - self.n_size + 1;
            let start: &str = &output[sidx..].join(" ");
            let next: &str = self.predict_next_token(start, &mut rng);
            output.push(next);
            if next == END_OF_STRING {
                return output.join(" ");
            }

            if max_tokens != 0 {
                count += 1;
            }
        }
    }

    fn predict_next_token(&self, context: &str, rng: &mut StdRng) -> &str {
        let candidates = match self.n_gram_map.get(context) {
            Some(c) => c,
            None => {
                return END_OF_STRING;
            }
        };

        let rand_float: f64 = rng.random();
        if self.top_k < 1.0 {
            let k_count = max(1, (candidates.len() as f64 * self.top_k) as usize);
            let shortened = &candidates[..k_count];
            let mut prob_sum: f64 = 0.0;
            for c in shortened.iter() {
                prob_sum += c.2;
            }
            let mut rolling_prob: f64 = 0.0;
            for c in shortened.iter() {
                rolling_prob += c.2 / prob_sum;
                if rand_float < rolling_prob {
                    return &c.0;
                }
            }
            return &candidates[0].0;
        }

        let mut rolling_prob = 0.0;
        for c in candidates.iter() {
            rolling_prob += c.2;
            if rand_float < rolling_prob {
                return &c.0;
            }
        }
        &candidates[0].0
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

        let model_json = serde_json::to_string(&model).expect("lol"); //TODO:Handle error
        match fs::write("model.json", &model_json) {
            Ok(_) => Ok(model_json),
            Err(err) => Err(format!("File write error: {}", err)),
        }
    }

    // TODO:Significant refactors needed
    // - model parameter and type has to be saved in the json
    // - change load be a constructor?
    fn load(&mut self, path: &str) -> Result<(), String> {
        self.n_gram_map = HashMap::new();
        self.start_tokens = Vec::new();
        let file = match File::open(path) {
            Ok(x) => x,
            Err(e) => return Err(format!("Failed to open file: {}", path)),
        };
        let reader = BufReader::new(file);
        let model_parsed: (Vec<String>, HashMap<String, Vec<(String, u32)>>) =
            match serde_json::from_reader(reader) {
                Ok(x) => x,
                Err(_) => return Err("Failed to parse model json file".to_owned()),
            };
        self.start_tokens = model_parsed.0;
        let model_shot = model_parsed.1;
        self.n_gram_map = model_shot
            .iter()
            .map(|(ctx, vec)| {
                let candidates = vec
                    .iter()
                    .map(|(s, u)| (s.clone(), *u, *u as f64 / vec.len() as f64))
                    .collect();
                (ctx.to_owned(), candidates)
            })
            .collect();

        println!("{}", serde_json::to_string(&self.n_gram_map).expect("lol")); //TODO:Handle error

        Ok(())
    }
}
