use simple_n_gram::*;

fn main() {
    println!("Hello, world!");

    let mut model = LidstoneModel::new(Config::default());
    model.load("model.json");

    // let config = Config::default();
    // let config = Config {
    //     n_size: 2,
    //     ..config
    // };
    // let mut model = LidstoneModel::new(config);
    // let mut first_pp = LowerCasePreProcessor::new();
    // let pp2 = StartEndTokensPreProcessor::new();
    // first_pp.set_next(Box::new(pp2));
    //
    // let raw_corpus: String = std::fs::read_to_string("corpus.txt").expect("Failed to read corpus");
    // let corpus_vector = raw_corpus.split("\n").collect::<Vec<&str>>();
    //
    // model
    //     .build_n_gram(Box::new(first_pp), corpus_vector)
    //     .unwrap();

    println!("Generating tokens...");
    println!("Done!\n\n{}", model.generate(0));
    println!("\n\n\n{}", model.generate(0));
    // match model.save() {
    //     Ok(_) => println!("Model saved!"),
    //     Err(err) => println!("{}", err),
    // }
}
