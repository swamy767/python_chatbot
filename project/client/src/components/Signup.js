import React, { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";
import '../components/Signup.css';

function Signup() {
    const navigate = useNavigate();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const passwordRegex = /^(?=.*\d)(?=.*[!@#$%^&*])(?=.*[a-z])(?=.*[A-Z]).{8,}$/;

    async function handleSubmit(e) {
        e.preventDefault();

        if (!passwordRegex.test(password)) {
            alert("Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.");
            return;
        }

        try {
            const response = await axios.post("http://localhost:8000/signup", {
                email,
                password
            });

            if (response.data === "exist") {
                alert("User already exists");
            } else if (response.data === "notexist") {
                navigate("/", { state: { id: email } });
            }
        } catch (error) {
            console.log(error);
            alert("Something went wrong");
        }
    }

    return (
        <div className="box2">
            <div className="signup">
                <h1>Signup</h1>
                <form onSubmit={handleSubmit}>
                    <input type="email" onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
                    <input type="password" onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
                    <button type="submit">Submit</button>
                </form>
                <br />
                <p>OR</p>
                <br />
                <Link to="/">Login Page</Link>
            </div>
        </div>
    );
}
export default Signup;
