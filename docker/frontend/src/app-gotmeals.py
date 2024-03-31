import datetime
import json
import os
from pathlib import Path

import requests
import streamlit as st

from elasticsearch_main import es, search_recipes

PROTOCOL = "https"
HOST = "edbrown.mids255.com"
PORT = 443

# streamlit run app.py

st.set_page_config(page_title="Got Meals?", page_icon="🍔")

#### SideBar ####

st.sidebar.title("What is Got Meals?")
st.sidebar.write(
    """
Our mission is unwavering – to remove the hassle from your daily culinary journey and bring joy back to your kitchen. With a simple photo, we unlock the potential of your ingredients and deliver a personalized list of recipes tailored to your preferences.

We created an easy to use and interactive application that allows your to take pictures of ingredients in the your fridge or cabinets and determines from those ingredients what recipes/meals you can make. This will provides real-time and instantaneous results and make cooking less of a hassle and more fun!

**Accuracy :** **Limitless**
                 
**Model :** **EfficientNet**

**Search supported by :** **ElasticSearch**

**Dataset :** **`?`**
"""
)

st.sidebar.markdown(
    "Created by **Ed Brown, Melissa Hartwick, Neha Jakkinpali, Melia Soque**"
)
st.sidebar.markdown(
    body="""

<th style="border:None"><a href="https://www.linkedin.com/in/jaydip-borad/" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="jaydipborad" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/boradj" target="blank"><img align="center" src="https://bit.ly/githubjd" alt="boradj" height="40" width="40" /></a></th>

""",
    unsafe_allow_html=True,
)


#### Main Body ####


def send_image_to_api(image_path, api_endpoint):
    if not os.path.exists(image_path):
        return {"error": "Image file does not exist."}
    files = {"file": open(image_path, "rb")}
    headers = {"accept": "application/json"}
    response = requests.post(api_endpoint, files=files, headers=headers)
    return response.json()


def main():
    st.title("Got Meals?🍔📷")
    st.header("Identify recipes based on the foods you have!")
    st.write(
        "Add your food images below and select any allergies to generate a list of relevant recipes"
    )

    uploaded_images = st.file_uploader(
        label="Upload a maximum of 5 items and a minimum of 1 item. Order the images you upload with the first image being the most important.",
        type=["jpg", "jpeg", "png", "heic"],
        accept_multiple_files=True,
    )

    option = st.selectbox(
        "Select from the dropdown list if any of these allergies apply to you",
        (
            "None",
            "Peanuts",
            "Tree Nuts",
            "All Nuts",
            "Milk",
            "Eggs",
            "Fish",
            "Shellfish",
            "Wheat",
            "Soybeans",
        ),
    )

    st.write("You selected:", option)
    responses = []  # Initialize a list to store responses for each image
    correct_names = (
        []
    )  # Initialize a list to store the correctness of ingredient names for each image
    ingredient_names = (
        []
    )  # Initialize a list to store ingredient names in the order of image uploads

    # Check if a file has been uploaded
    if uploaded_images is not None:
        num_images = len(uploaded_images)
        st.write(f"Number of Images Uploaded: {num_images}")

        for i in range(num_images):
            # Display the uploaded image
            st.image(
                uploaded_images[i],
                caption=f"Uploaded Image {i+1}",
                use_column_width=True,
            )

            # Check if the ingredient name is correct
            correct_name = st.radio(
                f"Is this the correct ingredient name for Image {i+1}?",
                ("Yes", "No"),
                key=f"radio_{i}",
            )
            correct_names.append(
                correct_name
            )  # Store the correctness of the ingredient name for each image

            # If the ingredient name is incorrect, provide a text input box to enter the correct name
            if correct_name == "No":
                corrected_name = st.text_input(
                    f"Please enter the correct ingredient name for Image {i+1}: ",
                    key=f"text_{i}",
                )
                st.write(f"Corrected name for Image {i+1}: {corrected_name}")
                ingredient_names.append(corrected_name)
            else:
                # Save the uploaded image to a temp directory
                temp_dir = "./temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_image_path = os.path.join(temp_dir, uploaded_images[i].name)
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_images[i].getvalue())

                # Send the image to the RESTAPI
                api_endpoint = f"{PROTOCOL}://{HOST}:{PORT}/predict"
                response = send_image_to_api(temp_image_path, api_endpoint)
                responses.append(response)  # Store the response for each image
                st.write(
                    f"Response for Image {i+1}: {response.get('ingredient', 'No ingredient found')}"
                )

                # If the ingredient name is correct, add it to the list of ingredient names
                ingredient_name = response.get("ingredient", "No ingredient found")
                ingredient_names.append(ingredient_name)

        # Display the submit button
        submit_button = st.button("Submit")

        if submit_button:
            # Ensure that the first ingredient takes priority in the recipe
            first_ingredient = ingredient_names[0]
            nice_to_have_ingredients = ingredient_names[1:]

            # Construct the list of ingredient names prioritized based on the order of image uploads
            ingredient_names_prioritized = [first_ingredient] + [
                ingredient
                for ingredient in nice_to_have_ingredients
                if ingredient != first_ingredient
            ]

            # Perform Elasticsearch query for recipes based on all ingredient names
            recipes = search_recipes(es, ingredient_names_prioritized)
            if recipes["hits"]["hits"]:
                for hit in recipes["hits"]["hits"]:
                    # Check if the recipe contains the selected allergy
                    if (
                        option != "None"
                        and option.lower().replace(" ", "_")
                        not in hit["_source"]["ingredients"].lower()
                    ):
                        st.write(f"Recipe Title: {hit['_source']['title']}")
                        st.write(f"Recipe Ingredients: {hit['_source']['ingredients']}")
                        st.write(f"Recipe Directions: {hit['_source']['directions']}")
                    elif option == "None":
                        st.write(f"Recipe Title: {hit['_source']['title']}")
                        st.write(f"Recipe Ingredients: {hit['_source']['ingredients']}")
                        st.write(f"Recipe Directions: {hit['_source']['directions']}")
            else:
                st.write("")


if __name__ == "__main__":
    main()
