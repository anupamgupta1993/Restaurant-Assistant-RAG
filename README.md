# Restaurant Assistant RAG ✨

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-lightgrey)](https://qdrant.ai/) [![Cursor IDE](https://img.shields.io/badge/Cursor_IDE-Editor-blue?logo=cursor&logoColor=white)](https://cursor.sh/)
 [![Grafana](https://img.shields.io/badge/Grafana-Monitoring-orange?logo=grafana&logoColor=white)](https://grafana.com/)


## 🧠 Project Overview

This is a RAG application to assist users with restaurant choices
### 🍔 Why Use the Uber Eats Dataset in a RAG Pipeline?

#### **1. Access to Real, Up-to-Date, Structured Data**

A plain **Large Language Model (LLM)**—even a strong one—only knows **general information** about restaurant menus. It *lacks* the specific, real-time data found in the Uber Eats dataset, such as:

* The **exact name** of a restaurant.
* Its **current menu items and prices**.
* **Location-specific cuisines**.
* **Rating or price ranges**.

A **Retrieval-Augmented Generation (RAG)** pipeline is essential here because it can retrieve **precise entries** directly from the dataset, allowing it to fulfill complex, specific queries like:

> *“Find all vegan options under $15 in New York”*

This approach ensures the generated answer is **accurate and current**, preventing the generation of vague or outdated information common to knowledge-cut LLMs.

#### **Use Cases**
- Menu price comparison across regions  
- Cuisine diversity visualization  
- Predictive modeling for menu pricing  
- Restaurant clustering or recommendation models  
- Correlation of menu pricing with socioeconomic indicators
## 🗂️ Dataset 

**Dataset:** [Uber Eats USA Restaurants & Menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus)  

### **Description**
This dataset was compiled via web-scraping of the Uber Eats platform’s US listings and captures detailed information about restaurants and their menus across the United States.  
It provides a rich basis for analyzing restaurant distribution, menu composition, pricing behavior, and regional trends in the food delivery ecosystem.


### **Contents & Structure**
The dataset consists of two main tables:

- **`restaurants.csv`** — ~63,000+ entries  
  Contains restaurant-level information:  
  *Restaurant ID, Name, Cuisine Type, Ratings, Price Range, Address, ZIP, Latitude/Longitude.*

- **`menu_items.csv`** — ~5 million+ entries  
  Contains menu item details linked to restaurants:  
  *Restaurant ID, Menu Category, Dish Name, Description, Price.*

## 🧰 Preparation

Since it uses OpenAI, you need to provide the API key:

1. Install `direnv`. If you use Ubuntu, run `sudo apt install direnv` and then `direnv hook bash >> ~/.bashrc`.
2. Copy `.envrc_template` into `.envrc` and insert your key there.
3. Run `direnv allow` to load the key into your environment.

For dependency management, i use pipenv, so you need to install it:

```bash
pip install pipenv
```

Once installed, you can install the app dependencies:

```bash
pipenv install --dev
```

## ⚡ Running the Application

### Running with Docker-Compose

The easiest way to run the application is with `docker-compose`:

```bash
docker-compose up
```

### Ingestion

The ingestion script is in [`ingest.py`](src/ingest.py).

Since i use a vector database, `Qdrant`, as the
knowledge base, i run the ingestion script at the startup
of the application.

It needs to be manually executed once to it reads and indexes the data inside qdrant which can take some time.

## 💻 Using the Application

When the application is running, you can start using it.

### CLI

I built an interactive CLI application using
[questionary](https://questionary.readthedocs.io/en/stable/).

To start it, run:

```bash
pipenv run python cli.py
```

You can also make it randomly select a question from the ground truth dataset (generated in retrieval evaulation notebook):

```bash
pipenv run python cli.py --random
```

### Using `requests`

When the application is running, you can use
[requests](https://requests.readthedocs.io/en/latest/)
to send questions—use test.py for testing it:

```bash
pipenv run python test.py
```

It will pick a random question from the ground truth dataset
and send it to the app.

### CURL

You can also use `curl` for interacting with the API:

```bash
URL=http://localhost:5001
QUESTION="Give me a list of highly rated italian restaraunts in new york under $$$ which serve meatball pasta"
DATA='{
    "question": "'${QUESTION}'"
}'

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${DATA}" \
    ${URL}/ask
```

You will see something like the following in the response:

```json
{
    "answer": "Restaurant Name: The Meatball Shop – Hell’s Kitchen
    Location: 798 9th Ave, New York, NY 10019
    Price Range: $$ (~$30)
    Signature Meatball Pasta Dish: Spaghetti in Meatballs (choice of tomato, parmesan cream, or pesto sauce)
    Why Go / Notes: Casual, fun, meatball-focused menu, popular for meatball cravings
    Rating / Reviews: 4.5+ / 1500+ reviews

    Restaurant Name: Pisticci
    Location: 125 La Salle St, New York, NY 10027
    Price Range: $$ (~$14–24 + meatball add-on)
    Signature Meatball Pasta Dish: Spaghetti and Meatballs (house meatballs optional)
    Why Go / Notes: Trattoria-style, relaxed, neighborhood vibe, good value
    Rating / Reviews: 4.5 / 1400+ reviews",
    "conversation_id": "4e1cef04-bfd9-4a2c-9cdd-2771d8f70e4d",
    "question": "Give me a list of highly rated italian restaraunts in new york under $$$ which serve meatball pasta"
}
```

Sending feedback:

```bash
ID="4e1cef04-bfd9-4a2c-9cdd-2771d8f70e4d"
URL=http://localhost:5001
FEEDBACK_DATA='{
    "conversation_id": "'${ID}'",
    "feedback": 1
}'

curl -X POST \
    -H "Content-Type: application/json" \
    -d "${FEEDBACK_DATA}" \
    ${URL}/feedback
```

After sending it, you'll receive the acknowledgement:

```json
{
    "message": "Feedback received for conversation 4e1cef04-bfd9-4a2c-9cdd-2771d8f70e4d: 1"
}
```