from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps

# init
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)


# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')


# reguired toekn decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({
                "status": 401,
                "success": False,
                "message": "Token is missing"
            }), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.filter_by(id=data['user_id']).first()
        except Exception as e:
            print(e)
            return jsonify({
                "status": 401,
                "success": False,
                "message": "Token is invalid"
            }), 401
        return f(current_user, *args, **kwargs)

    return decorated


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({
            "status": 400,
            "success": False,
            "message": "Username already exists"
        }), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({
            "status": 400,
            "success": False,
            "message": "Email already exists"
        }), 400

    hashed_password = generate_password_hash(data['password'])
    new_user = User(
        username=data['username'],
        password=hashed_password,
        email=data['email'],
        role='user'
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({
        "status": 201,
        "success": True,
        "message": "User registered successfully",
        "data": {
            "username": new_user.username,
            "email": new_user.email,
            "role": new_user.role
        }
    }), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({
            "status": 401,
            "success": False,
            "message": "Invalid username or password"
        }), 401

    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        "status": 200,
        "success": True,
        "message": "Login successful",
        "token": token,
        "data": {
            "username": user.username,
            "email": user.email,
            "role": user.role
        }
    }), 200


@app.route('/api/users', methods=['GET'])
@token_required
def get_users(current_user):
    all_users = User.query.all()
    output = []
    for user in all_users:
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role
        }
        output.append(user_data)
    return jsonify({
        "status": 200,
        "success": True,
        "data": output
    }), 200


@app.route('/api/health')
def health_check():
    return jsonify({
        "status": 200,
        "success": True,
        "message": "API is healthy"
    }), 200


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=3000)
