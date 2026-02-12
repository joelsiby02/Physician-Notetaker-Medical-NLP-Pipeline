from src.pipeline import run_pipeline, save_outputs


def main():
    with open("data/sample_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()

    results = run_pipeline(transcript)
    save_outputs(results)

    print("Done. Outputs saved to /outputs")
    print("\nStructured Summary Preview:\n")
    print(results["structured_summary"])
    

if __name__ == "__main__":
    
    main()
