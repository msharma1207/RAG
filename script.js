function sendMessage() {
    let userInput = document.getElementById("user_input").value;
    if (!userInput) return;

    let chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += `<p><b>You:</b> ${userInput}</p>`;

    fetch("/chat", {
        method: "POST",
        body: new URLSearchParams({ user_input: userInput }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    })
    .then(response => response.json())
    .then(data => {
        chatbox.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
        document.getElementById("user_input").value = "";
    })
    .catch(error => console.error("Error:", error));
}
