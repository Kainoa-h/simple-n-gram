use simple_n_gram::*;

fn main() {
    println!("Hello, world!");

    let model_from_file = LidstoneModel::load("model.json").expect("lol");

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
    //
    // println!("Generating tokens...");
    // println!("Model:\n\n{}", model.generate(0));
    let seed = 2314235;
    println!("Model from file:\n\n{}", model_from_file.generate(0, seed));
    println!("Model from file:\n\n{}", model_from_file.generate(0, seed));

    println!("Ready!");

    loop {
        println!("\n\nEnter a seed:");
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).expect("AAHH");
        let value: String = line.parse().expect("OI");
        let value: &str = value.trim();

        println!(
            "\n\n\n{}",
            model_from_file.generate(0, value.parse().expect("numba"))
        );
    }

    // println!("Done!\n\n{}", model.generate(0));
    // println!("\n\n\n{}", model.generate(0));
    // match model.save() {
    //     Ok(_) => println!("Model saved!"),
    //     Err(err) => println!("{}", err),
    // }
}
