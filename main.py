from aggregator_agent import aggregator_agent

def main():
    print("Welcome to the Multi-Agent Financial System.")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        query = input("\nEnter your financial query or request: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting the system. Goodbye!")
            break

        # Call the aggregator agent to process the query.
        # print_response() streams the output to the console.
        aggregator_agent.print_response(query, max_tokens=1000)

if __name__ == "__main__":
    main()
