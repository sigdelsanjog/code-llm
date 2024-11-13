import './ChatBox.css'

import React, { useState } from "react";
import axios from "axios";

const ChatBox = () => {
  const [prompt, setPrompt] = useState(""); // Holds the user input
  const [response, setResponse] = useState(""); // Holds the API response

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResponse(""); // Clear previous response

    try {
      // Send the prompt to the backend FastAPI
      const result = await axios.post("http://localhost:8000/generate-code", { prompt });

      // Log the API response (optional for debugging)
      console.log("API Response:", result.data.generated_code);

      // Set the response state with the data from the backend
      setResponse(result.data.generated_code); // Updated to access 'generated_code'
      console.log("Response state updated:", result.data.generated_code); // Debug log
    } catch (error) {
      console.error("Error fetching response:", error);
      setResponse("There was an error processing your request.");
    }
  };

  return (
    <div className="chat-container">
      {/* Input form for the prompt */}
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter your prompt here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)} // Update prompt as user types
        />
        <button type="submit">Submit</button>
      </form>

      {/* Display the API response */}
      <div className="response">
        {response ? (
          <p>{response}</p> // Render the API response inside a <p> tag
        ) : (
          <p>No response yet. Please submit a prompt.</p> // Show default message when no response
        )}
      </div>
    </div>
  );
};

export default ChatBox;
