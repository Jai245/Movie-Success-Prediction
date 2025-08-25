import gradio as gr
import pickle

# Load your trained model
with open("hit_flop_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict_movie_success(budget, genre, cast, director, runtime):
<<<<<<< HEAD
    # Keep categorical inputs as strings, numeric inputs as numbers
=======
    # Create input features (adjust to match how you trained your model)
>>>>>>> ddf82ea3301ee4c2df1d2832eef8a6752ebc3169
    features = [[budget, genre, cast, director, runtime]]
    prediction = model.predict(features)[0]
    return "Hit 🎬" if prediction == 1 else "Flop ❌"

# Gradio interface
demo = gr.Interface(
    fn=predict_movie_success,
    inputs=[
        gr.Number(label="Budget (in millions)"),
        gr.Textbox(label="Genre"),
        gr.Textbox(label="Cast"),
        gr.Textbox(label="Director"),
        gr.Number(label="Runtime (minutes)")
    ],
    outputs="text",
    title="🎥 Movie Success Prediction",
    description="Enter movie details to predict if it will be a Hit or Flop"
)

if __name__ == "__main__":
    demo.launch()
