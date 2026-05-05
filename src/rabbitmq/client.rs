pub mod client;



use rabbitmq_stream_client::{
    Environment,
    error::StreamCreateError,
    types::{ByteCapacity, Message, ResponseCode},
};

use std::env;



async fn connect() -> Result<(), Box<dyn std::error::Error>> {

    let host = env::var("RABBITMQ_HOST").unwrap_or_else(|_| "localhost".to_string());
    let port = env::var("RABBITMQ_PORT").unwrap_or_else(|_| "5552".to_string());
    let user = env::var("RABBITMQ_USER").expect("RABBITMQ_USER must be set");
    let pass = env::var("RABBITMQ_PASS").expect("RABBITMQ_PASS must be set");


    let environment = Environment::builder()
        .host(host)
        .port(port)
        .username(user)
        .password(pass)
        .build()
        .await?;

    let stream = "hello-rust-stream";

    let create_response = environment
        .stream_creator()
        .max_length(ByteCapacity::GB(5))
        .create(stream)
        .await;

    if let Err(e) = create_response {
        if let StreamCreateError::Create { stream, status } = e {
            match status {
                ResponseCode::StreamAlreadyExists => {}
                err => panic!("Error creating stream: {:?} {:?}", stream, err),
            }
        }
    }

    let producer = environment.producer().build(stream).await?;

    producer
        .send_with_confirm(Message::builder().body("Hello, World!").build())
        .await?;

    println!("Message sent!");

    let consumer = environment
        .consumer()
        .build(stream)
        .await?;

    // NOTE(nasr): this loops forever
    loop {
        match consumer.next().await {
            Ok(delivery) => {
                let data = delivery.message().data().unwrap_or_default();
                println!("Received: {}", String::from_utf8_lossy(data));
            }
            Err(e) => {
                eprintln!("Consumer error: {:?}", e);
                break;
            }
        }
    }

    Ok(())
}
