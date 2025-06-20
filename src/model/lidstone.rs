use crate::{END_OF_STRING, Model};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use std::fs::{self, File};
use std::io::BufReader;
use std::num;
use std::{
    cmp::max,
    collections::{HashMap, HashSet},
};

pub struct LidstoneConfig {
    pub n_size: usize,
    pub top_k: f64,
    pub temperature: f64,
}

impl Default for LidstoneConfig {
    fn default() -> Self {
        Self {
            n_size: 2,
            top_k: 1.0,
            temperature: 1.0,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct LidstoneModel {
    n_size: usize,
    top_k: f64,
    temperature: f64,
    vocabulary_array: Vec<String>,
    #[serde(serialize_with = "serialize_n_gram_map")]
    n_gram_map: HashMap<String, Vec<(String, u32, f64)>>,
    start_tokens: Vec<String>,
}

fn serialize_n_gram_map<S>(
    map: &HashMap<String, Vec<(String, u32, f64)>>, serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut s_map = serializer.serialize_map(Some(map.len()))?;
    for (key, vec_of_tuples) in map {
        let mut temp_vec = Vec::new();
        for (s, u, _) in vec_of_tuples {
            temp_vec.push((s, u, 0.0f64));
        }
        s_map.serialize_entry(key, &temp_vec)?;
    }
    s_map.end()
}

impl LidstoneModel {
    pub fn new(config: LidstoneConfig) -> Self {
        Self {
            n_size: config.n_size,
            top_k: config.top_k,
            temperature: config.temperature,
            vocabulary_array: Vec::new(),
            n_gram_map: HashMap::new(),
            start_tokens: Vec::new(),
        }
    }
}

impl Model for LidstoneModel {
    type ModelError = String;

    fn build_n_gram<P>(&mut self, pre_processor_chain: P, corpus: Vec<&str>) -> Result<(), String>
    where
        P: Fn(String) -> String,
    {
        let mut n_gram_map_builder: HashMap<String, HashMap<String, u32>> = HashMap::new();
        let mut vocab_set: HashSet<String> = HashSet::new();

        for (idx, &sentence) in corpus.iter().enumerate() {
            if sentence.is_empty() {
                continue;
            };
            //TODO:Lots of copying here...
            let tokenized_sent = pre_processor_chain(sentence.to_owned())
                .split_whitespace()
                .map(|x| x.to_owned())
                .collect::<Vec<String>>();

            if tokenized_sent.len() < self.n_size {
                return Err(format!(
                    "Length of sentence at line {} is shorter than ngram size of {}:\n\"{}\"",
                    idx, self.n_size, sentence
                ));
            }

            vocab_set.extend(tokenized_sent.iter().cloned());
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
        //TODO: is there a way to do this without having to copy the string?
        self.vocabulary_array = vocab_set
            .iter()
            .map(|x| x.to_owned())
            .collect::<Vec<String>>();
        self.vocabulary_array.sort();

        Ok(())
    }

    fn generate(&self, max_tokens: u32, seed: u64) -> String {
        let mut rng = StdRng::seed_from_u64(seed);
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
                return &self.vocabulary_array[rng.random_range(0..self.start_tokens.len())];
            }
        };

        let rand_float: f64 = rng.random();

        if self.top_k < 1.0 && self.temperature != 1.0 {
            let one_over_temp = 1f64 / self.temperature;
            let k_count = max(1, (candidates.len() as f64 * self.top_k) as usize);
            let shortened = &candidates[..k_count];

            let mut prob_sum: f64 = 0.0;
            for c in shortened.iter() {
                prob_sum += c.2.powf(one_over_temp);
            }
            let mut rolling_prob: f64 = 0.0;
            for c in shortened.iter() {
                rolling_prob += c.2.powf(one_over_temp) / prob_sum;
                if rand_float < rolling_prob {
                    return &c.0;
                }
            }
            return &candidates[0].0;
        }

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
        let model_json = serde_json::to_string(&self).expect("lol"); //TODO:Handle error
        match fs::write("model.json", &model_json) {
            Ok(_) => Ok(model_json),
            Err(err) => Err(format!("File write error: {}", err)),
        }
    }

    fn load(path: &str) -> Result<LidstoneModel, String> {
        let file = match File::open(path) {
            Ok(x) => x,
            Err(_) => return Err(format!("Failed to open file: {}", path)),
        };
        let reader = BufReader::new(file);

        let mut model_parsed: LidstoneModel = match serde_json::from_reader(reader) {
            Ok(x) => x,
            Err(_) => return Err("Failed to parse the model file".to_owned()),
        };

        model_parsed.n_gram_map.iter_mut().for_each(|(_ctx, vec)| {
            let total: f64 = vec.iter().map(|x| x.1).sum::<u32>() as f64;
            vec.iter_mut().for_each(|(_s, u, f)| {
                *f = *u as f64 / total;
            });
        });

        Ok(model_parsed)
    }
}
