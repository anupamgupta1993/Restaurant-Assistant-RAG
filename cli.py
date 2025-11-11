import json
import uuid
import argparse

import requests
import questionary

import pandas as pd


def get_random_question(file_path):
    df = pd.read_csv(file_path)
    return df.sample(n=1).iloc[0]["question"]


def ask_question(url, question):
    data = {"question": question}
    response = requests.post(url, json=data)
    
    # Check if request was successful
    response.raise_for_status()
    
    # Check if response contains JSON
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Server returned non-JSON response. Status: {response.status_code}")
        print(f"Response text: {response.text[:500]}")
        raise


def send_feedback(url, conversation_id, feedback):
    feedback_data = {"conversation_id": conversation_id, "feedback": feedback}
    response = requests.post(f"{url}/feedback", json=feedback_data)
    response.raise_for_status()
    return response.status_code


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI app for continuous question answering and feedback"
    )
    parser.add_argument(
        "--random", action="store_true", help="Use random questions from the CSV file"
    )
    args = parser.parse_args()

    base_url = "http://localhost:5001"
    csv_file = "./data/ground-truth-retrieval.csv"

    print("Welcome to the interactive question-answering app!")
    print("You can exit the program at any time when prompted.")

    while True:
        if args.random:
            question = get_random_question(csv_file)
            print(f"\nRandom question: {question}")
        else:
            question = questionary.text("Enter your question:").ask()

        if not question:
            print("No question provided. Exiting.")
            break

        try:
            response = ask_question(f"{base_url}/ask", question)
            
            if "error" in response:
                print(f"\nError: {response['error']}")
                continue
            
            print("\nAnswer:", response.get("answer", "No answer provided"))
            
            conversation_id = response.get("conversation_id", str(uuid.uuid4()))

            feedback = questionary.select(
                "How would you rate this response?",
                choices=["+1 (Positive)", "-1 (Negative)", "Pass (Skip feedback)"],
            ).ask()

            if feedback != "Pass (Skip feedback)":
                feedback_value = 1 if feedback == "+1 (Positive)" else -1
                try:
                    status = send_feedback(base_url, conversation_id, feedback_value)
                    print(f"Feedback sent. Status code: {status}")
                except requests.exceptions.RequestException as e:
                    print(f"Error sending feedback: {e}")
            else:
                print("Feedback skipped.")

        except requests.exceptions.RequestException as e:
            print(f"\nError communicating with server: {e}")
            print(f"Make sure the server is running at {base_url}")
            continue
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            continue

        continue_prompt = questionary.confirm("Do you want to continue?").ask()
        if not continue_prompt:
            print("Thank you for using the app. Goodbye!")
            break


if __name__ == "__main__":
    main()