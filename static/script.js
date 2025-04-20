document.getElementById("sendButton").addEventListener("click", async () => {
    const userInput = document.getElementById("userInput").value.trim();
    if (!userInput) {
        alert("Please enter a question!");
        return;
    }

    try {
        // Send the user query to the backend
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput }),
        });

        if (!response.ok) {
            throw new Error("Failed to get a response from the chatbot");
        }

        const result = await response.json();
        const messages = document.getElementById("messages");

        // Display the user query and chatbot response
        messages.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;
        messages.innerHTML += `<div><strong>Bot:</strong> ${result.response}</div>`;
        document.getElementById("userInput").value = ""; 
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
});