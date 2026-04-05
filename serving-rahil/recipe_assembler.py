"""
serving-rahil/recipe_assembler.py

Assembles a Mealie-compatible draft recipe from an EfficientNet-B0 prediction.

Production flow:
  1. EfficientNet-B0 predicts food category from image
  2. Category is used to query ANN index built on Recipe1M (data pipeline)
  3. Top-K nearest neighbor recipes are retrieved
  4. Best match is assembled into Mealie JSON schema

Current state (Initial Implementation):
  ANN retrieval is mocked with hardcoded representative templates per food category.
  When ANN index is ready, replace mock_ann_retrieval() with a real query.

Mealie recipe schema reference:
  https://hay-kot.github.io/mealie/api/docs (POST /api/recipes)

"""

from __future__ import annotations

# Food category templates (mock ANN retrieval from Recipe1M) 
# Each entry maps an ImageNet food class → representative Mealie recipe draft.
# In production, these are replaced by the top-1 result from the ANN index.

_TEMPLATES: dict[str, dict] = {
    "pizza": {
        "name": "Classic Margherita Pizza",
        "description": "A simple Italian pizza with fresh tomatoes and mozzarella.",
        "recipeCategory": "Italian",
        "tags": ["pizza", "italian", "vegetarian"],
        "recipeIngredient": [
            "2 cups all-purpose flour",
            "1 tsp active dry yeast",
            "1 cup tomato sauce",
            "200g fresh mozzarella",
            "Fresh basil leaves",
            "2 tbsp olive oil",
            "Salt to taste",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Mix flour, yeast, and salt. Add water and knead into dough."},
            {"id": "step_2", "text": "Let dough rise for 1 hour covered."},
            {"id": "step_3", "text": "Spread tomato sauce on rolled-out dough."},
            {"id": "step_4", "text": "Add mozzarella and bake at 475F for 10-12 minutes."},
            {"id": "step_5", "text": "Top with fresh basil and drizzle with olive oil."},
        ],
        "allergens": ["gluten", "dairy"],
    },
    "burger": {
        "name": "Classic Beef Burger",
        "description": "A juicy homemade beef burger with fresh toppings.",
        "recipeCategory": "American",
        "tags": ["burger", "beef", "american"],
        "recipeIngredient": [
            "500g ground beef (80/20 lean)",
            "4 burger buns",
            "4 slices cheddar cheese",
            "Lettuce, tomato, red onion",
            "Salt and black pepper",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Season ground beef with salt and pepper, form into patties."},
            {"id": "step_2", "text": "Grill patties 4 minutes per side for medium doneness."},
            {"id": "step_3", "text": "Add cheese in the last minute of cooking."},
            {"id": "step_4", "text": "Assemble with toppings and serve immediately."},
        ],
        "allergens": ["gluten", "dairy"],
    },
    "pasta": {
        "name": "Spaghetti Carbonara",
        "description": "Classic Roman pasta with eggs, aged cheese, and pancetta.",
        "recipeCategory": "Italian",
        "tags": ["pasta", "italian", "carbonara"],
        "recipeIngredient": [
            "400g spaghetti",
            "200g pancetta or guanciale",
            "4 large eggs",
            "100g Pecorino Romano, finely grated",
            "Freshly cracked black pepper",
            "Salt for pasta water",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Cook spaghetti in heavily salted boiling water until al dente."},
            {"id": "step_2", "text": "Fry pancetta in a cold pan until fat renders and edges crisp."},
            {"id": "step_3", "text": "Whisk eggs with grated cheese and generous black pepper."},
            {"id": "step_4", "text": "Drain pasta, reserve 1 cup pasta water."},
            {"id": "step_5", "text": "Toss hot pasta with pancetta off heat, pour egg mixture, toss quickly adding pasta water to emulsify."},
        ],
        "allergens": ["gluten", "dairy", "eggs"],
    },
    "soup": {
        "name": "Classic Tomato Soup",
        "description": "Rich and warming tomato soup from roasted fresh tomatoes.",
        "recipeCategory": "Comfort Food",
        "tags": ["soup", "tomato", "vegetarian", "comfort-food"],
        "recipeIngredient": [
            "800g ripe tomatoes, halved",
            "1 large onion, quartered",
            "4 cloves garlic",
            "2 tbsp olive oil",
            "1 cup vegetable broth",
            "1 tsp sugar",
            "Salt and black pepper",
            "Fresh basil to serve",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Roast tomatoes, onion, and garlic with olive oil at 400F for 30 minutes."},
            {"id": "step_2", "text": "Transfer to pot, add broth and sugar, simmer 10 minutes."},
            {"id": "step_3", "text": "Blend until smooth using immersion blender."},
            {"id": "step_4", "text": "Season with salt and pepper, serve with fresh basil."},
        ],
        "allergens": [],
    },
    "dessert": {
        "name": "Chocolate Lava Cake",
        "description": "Individual chocolate cake with a warm molten center.",
        "recipeCategory": "Dessert",
        "tags": ["dessert", "chocolate", "baking"],
        "recipeIngredient": [
            "100g dark chocolate (70%)",
            "100g unsalted butter",
            "2 whole eggs plus 2 egg yolks",
            "80g powdered sugar",
            "30g all-purpose flour",
            "Pinch of salt",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Melt chocolate and butter together over a double boiler."},
            {"id": "step_2", "text": "Whisk eggs, yolks, and sugar until pale and slightly thickened."},
            {"id": "step_3", "text": "Fold chocolate into egg mixture, sift in flour, fold gently."},
            {"id": "step_4", "text": "Pour into buttered ramekins. Bake at 425F for 12 minutes."},
            {"id": "step_5", "text": "Invert onto plate immediately and serve with ice cream."},
        ],
        "allergens": ["gluten", "dairy", "eggs"],
    },
    "salad": {
        "name": "Mediterranean Chickpea Salad",
        "description": "A fresh and hearty salad with chickpeas, vegetables, and feta.",
        "recipeCategory": "Mediterranean",
        "tags": ["salad", "vegetarian", "healthy", "mediterranean"],
        "recipeIngredient": [
            "400g canned chickpeas, drained",
            "1 cucumber, diced",
            "2 cups cherry tomatoes, halved",
            "100g feta cheese, crumbled",
            "1/4 red onion, thinly sliced",
            "3 tbsp olive oil",
            "2 tbsp lemon juice",
            "Fresh parsley and mint",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Combine chickpeas, cucumber, tomatoes, and onion in a large bowl."},
            {"id": "step_2", "text": "Whisk olive oil, lemon juice, salt and pepper for dressing."},
            {"id": "step_3", "text": "Toss salad with dressing, top with feta and fresh herbs."},
        ],
        "allergens": ["dairy"],
    },
    "default": {
        "name": "Mixed Vegetable Stir Fry",
        "description": "A healthy and colorful vegetable stir fry with a savory sauce.",
        "recipeCategory": "Asian",
        "tags": ["stir-fry", "vegetarian", "healthy", "quick"],
        "recipeIngredient": [
            "2 cups mixed vegetables (broccoli, bell pepper, snap peas, carrots)",
            "2 tbsp soy sauce",
            "1 tbsp oyster sauce",
            "1 tsp sesame oil",
            "2 cloves garlic, minced",
            "1 tsp fresh ginger, grated",
            "1 tbsp vegetable oil",
        ],
        "recipeInstructions": [
            {"id": "step_1", "text": "Heat oil in wok over high heat until smoking."},
            {"id": "step_2", "text": "Add garlic and ginger, stir fry 30 seconds."},
            {"id": "step_3", "text": "Add harder vegetables first, then softer ones, stir fry 4-5 minutes."},
            {"id": "step_4", "text": "Add sauces, toss to coat, finish with sesame oil. Serve over rice."},
        ],
        "allergens": ["soy", "shellfish"],
    },
}

# ── ImageNet class index → food category mapping ───────────────────────────────
# EfficientNet-B0 trained on ImageNet 1000 classes.
# These indices correspond to food-related classes in ImageNet label set.
FOOD_CLASS_MAP: dict[int, str] = {
    924: "guacamole",   925: "soup",        926: "dessert",
    927: "dessert",     928: "dessert",     929: "bread",
    930: "bread",       931: "bread",       932: "bread",
    933: "burger",      934: "hotdog",      935: "pasta",
    936: "salad",       937: "vegetables",  938: "vegetables",
    939: "vegetables",  940: "vegetables",  941: "vegetables",
    942: "vegetables",  943: "vegetables",  944: "vegetables",
    945: "vegetables",  946: "mushroom",    947: "mushroom",
    948: "fruit",       949: "fruit",       950: "fruit",
    951: "fruit",       952: "fruit",       953: "fruit",
    954: "fruit",       955: "fruit",       956: "fruit",
    957: "fruit",       958: "meat",        959: "pasta",
    960: "dessert",     961: "dessert",     962: "dessert",
    963: "pizza",       964: "pie",         965: "mexican",
    966: "beverage",    967: "beverage",    968: "beverage",
}


def mock_ann_retrieval(category: str) -> dict:
    """
    Mock ANN retrieval from Recipe1M index.

    Production replacement:
        query = embed_category(category)            # encode category as vector
        results = ann_index.search(query, top_k=5)  # FAISS/ScaNN index
        return results[0]                           # best match recipe

    Args:
        category: Predicted food category string (e.g. "pizza", "burger")

    Returns:
        Recipe template dict matching Mealie schema
    """
    return _TEMPLATES.get(category, _TEMPLATES["default"])


def assemble_draft(
    class_idx: int,
    confidence: float,
    bypass_gate: bool = False,
) -> tuple[str, dict]:
    """
    Assemble a Mealie-compatible draft recipe from model prediction.

    Args:
        class_idx:   ImageNet class index from EfficientNet-B0
        confidence:  Softmax probability of top prediction
        bypass_gate: If True, skip food class check (for benchmarking)

    Returns:
        (category, draft_recipe_dict)
    """
    category = FOOD_CLASS_MAP.get(class_idx, "default") if not bypass_gate else "default"
    template = mock_ann_retrieval(category)

    draft = {k: v for k, v in template.items() if k != "allergens"}
    draft["nutrition"] = {
        "disclaimer": (
            "AI-generated nutritional information is approximate. "
            "Please verify all allergens independently before serving."
        ),
        "allergens": template.get("allergens", []),
    }
    draft["retrieval_metadata"] = {
        "source_dataset": "Recipe1M",
        "retrieval_method": "mock_ann (production: FAISS index)",
        "top_k_candidates": 5,
        "selected_rank": 1,
        "similarity_score": round(float(confidence), 4),
        "food_category": category,
        "imagenet_class_idx": class_idx,
        "model": "efficientnet-b0-imagenet-v1",
    }
    return category, draft
