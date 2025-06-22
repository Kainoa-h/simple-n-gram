use simple_n_gram::*;

fn main() {
    const DEFAULT_MODEL_PATH: &str = "model.json";
    const DEFUALT_CORPUS_PATH: &str = "corpus.txt";
    const MODEL_TEMP: f64 = 1.0;
    const MODEL_TOP_K: f64 = 1.0;
    let mut model: LidstoneModel;

    //init
    loop {
        println!("Initialise a Model!");
        println!("0 - Exit");
        println!("1 - Load model from {}", DEFAULT_MODEL_PATH);
        println!("2 - Build model from {}", DEFUALT_CORPUS_PATH);

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).expect("AAHH");
        let value: String = line.parse().expect("OI");
        let value: &str = value.trim();

        match value {
            "0" => {
                panic!("lol");
            }
            "1" => {
                model = LidstoneModel::load(DEFAULT_MODEL_PATH).expect("lol");
                break;
            }
            "2" => {
                let config = LidstoneConfig { n_size: 3 };
                model = LidstoneModel::new(config);

                let raw_corpus: String =
                    std::fs::read_to_string(DEFUALT_CORPUS_PATH).expect("Failed to read corpus");
                let corpus_vector = raw_corpus.split("\n").collect::<Vec<&str>>();

                model
                    .build_n_gram(
                        |s| {
                            PreProcessor::new(s)
                                .lowercase()
                                .add_start_end_tokens()
                                .done()
                        },
                        corpus_vector,
                    )
                    .expect("Model failed to build");
                break;
            }
            _ => {
                println!("\n\n");
                continue;
            }
        }
    }
    println!("Ready!\n (0 - stop)");

    loop {
        println!("\n\nEnter a seed:");
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).expect("AAHH");
        let value: String = line.parse().expect("OI");
        let value: &str = value.trim();
        if value == "0" {
            break;
        }

        println!(
            "\n\n\n{}",
            model.generate(0, value.parse().expect("numba"), MODEL_TOP_K, MODEL_TEMP)
        );
    }

    println!("Save model? (y/n):");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("AAHH");
    let value: String = line.parse().expect("OI");
    let value: &str = value.trim();
    if value == "y" {
        match model.save() {
            Ok(_) => println!("Model saved!"),
            Err(err) => println!("{}", err),
        }
    }
}
