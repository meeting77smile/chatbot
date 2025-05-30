use anyhow::Result;
use clap::Parser;
mod cli;
mod database;
mod embeddings;
mod llm;

#[tokio::main]
async fn main() -> Result<()> {
    let args = cli::Cli::parse();
    
    match args.command {
        cli::Commands::Ask { query } => {
            // 1. 获取相关上下文
            let references = database::retrieve(&query).await?;
            
            // 2. 使用LLM生成回答
            let answer = llm::answer_with_context(&query, references).await?;
            
            println!("Answer: {}", answer);
        }
        cli::Commands::Remember { content } => {
            // 存储内容到知识库
            database::insert(&content).await?;
            println!("I've remembered that!");
        }
    }
    
    Ok(())
}