
# %%
#########################################
# flask
#########################################

from flask import Flask, redirect, request, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# %%
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
login_manager = LoginManager()

# %%
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = "secret554"
# %%
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "login_page"
# %%
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
# %%

class User(db.Model, UserMixin):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    email: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email
        }
# %%
@app.route("/")
@login_required
def main_page():
    return render_template("main.html")

# %%
@app.route("/contact")
def contact_page():
    return render_template("contact.html")
# %%

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        values = request.form
        user = User.query.filter_by(email=values["email"]).first()
        if user and check_password_hash(user.password, values["password"]):
            login_user(user)
            return redirect("/")
        else:
            return "Invalid credentials"
    return render_template("login.html")
# %%
@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        values = request.form
        
        password = generate_password_hash(values["password"])
        
        new_user = User(
            email=values["email"],
            username=values["email"],
            password=password
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")
# %%
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/main")

# %%
@app.route("/profile")
@login_required
def profile_page():
    return render_template("profile.html")


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_test_norm = x_test.astype('float32') / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_simple = tf.keras.models.load_model('model/simple_model.keras')
model_complex = tf.keras.models.load_model('model/complex_model.keras')

#########


@app.route('/predict', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        
        indices = [
            request.form.get('idx1', type=int),
            request.form.get('idx2', type=int),
            request.form.get('idx3', type=int)
        ]

        for i, idx in enumerate(indices):
            if idx is not None and 0 <= idx < len(x_test):
                img = x_test[idx]
                img_input = x_test_norm[idx].reshape(1, 28, 28) 

                pred_simple = model_simple.predict(img_input)
                label_simple = class_names[np.argmax(pred_simple)]
                pred_complex = model_complex.predict(img_input)
                label_complex = class_names[np.argmax(pred_complex)]

                true_label = class_names[y_test[idx]]
                
                img_path = f'static/temp/img_{idx}.png'
                plt.imsave(img_path, img, cmap='gray')

                results.append({
                    'id': idx,
                    'img_path': img_path,
                    'true_label': true_label,
                    'pred_simple': label_simple,
                    'pred_complex': label_complex
                })

    return render_template('predict.html', results=results)
#########

# %%
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(
        "127.0.0.1",
        5001,
        debug=True
    )
# %%