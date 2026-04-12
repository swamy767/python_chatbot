
import React from "react"
import {useLocation, useNavigate} from 'react-router-dom';

function Home (){
    const location=useLocation()

    return (
        <div className="homepage">

            <h1>Hello {location.state.id} and welcome to the home</h1>

        </div>
    )
}
function showOptions() {
    let category = document.getElementById("category").value;
    let subDiv = document.getElementById("subOptions");

    let options = {
        ece: ["Overview", "Subjects", "Career", "Projects", "Overall"],
        fees: ["Tuition Fees", "Hostel Fees", "Transport", "Overall"],
        hostel: ["HMG Hostel", "Rajagruha Hostel", "Facilities", "Overall"]
    };

    subDiv.innerHTML = "";

    // ✅ Fix: check if category exists
    if (!options[category]) return;

    options[category].forEach(opt => {
        let btn = document.createElement("button");
        btn.innerText = opt;
        btn.onclick = () => showAnswer(category, opt);
        subDiv.appendChild(btn);
    });
}
export default Home
