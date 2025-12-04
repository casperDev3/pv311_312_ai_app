from flask import Flask, jsonify, request, render_template
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
    created_user_id = User.query.filter_by(username=data['username']).first().id

    token = jwt.encode({
        'user_id': created_user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        "status": 201,
        "success": True,
        "message": "User registered successfully",
        "data": {
            "username": new_user.username,
            "email": new_user.email,
            "role": new_user.role,
            "token": token
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


@app.route('/api/users/<int:user_id>', methods=['GET'])
@token_required
def get_user(current_user, user_id):
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({
            "status": 404,
            "success": False,
            "message": "User not found"
        }), 404

    if current_user.id != user.id and current_user.role != 'admin':
        return jsonify({
            "status": 403,
            "success": False,
            "message": "You do not have permission to view this user"
        }), 403

    user_data = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role
    }

    return jsonify({
        "status": 200,
        "success": True,
        "data": user_data
    }), 200


@app.route('/api/users/<int:user_id>', methods=['PUT'])
@token_required
def update_user(current_user, user_id):
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({
            "status": 404,
            "success": False,
            "message": "User not found"
        }), 404

    if current_user.id != user.id and current_user.role != 'admin':
        return jsonify({
            "status": 403,
            "success": False,
            "message": "You do not have permission to edit this user"
        }), 403

    data = request.get_json()
    if 'username' in data:
        user.username = data['username']
    if 'email' in data:
        user.email = data['email']
        # if 'password' in data:
        #     user.password = generate_password_hash(data['password'])

    if 'role' in data and current_user.role == 'admin':
        user.role = data['role']
    else:
        return jsonify({
            "status": 403,
            "success": False,
            "message": "Only admin can change user roles"
        }), 403

    db.session.commit()
    return jsonify({
        "status": 200,
        "success": True,
        "message": "User updated successfully",
        "data": {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role
        }
    }), 200


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@token_required
def delete_user(current_user, user_id):
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({
            "status": 404,
            "success": False,
            "message": "User not found"
        }), 404

    if current_user.id != user.id and current_user.role != 'admin':
        return jsonify({
            "status": 403,
            "success": False,
            "message": "You do not have permission to delete this user"
        }), 403

    db.session.delete(user)
    db.session.commit()
    return jsonify({
        "status": 200,
        "success": True,
        "message": "User deleted successfully"
    }), 200


# default routes
@app.route('/')
def index():
    return render_template("index.html", title="User Management API")


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
