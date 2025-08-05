import './ChatBox.css'

import React, { useState } from "react";
import axios from "axios";

const ChatBox = () => {
  const [prompt, setPrompt] = useState(""); // Holds the user input
  const [response, setResponse] = useState(null); // Holds the API response object
  const [loading, setLoading] = useState(false); // Loading state

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResponse(null); // Clear previous response
    setLoading(true); // Set loading state

    try {
      // Send the prompt to the backend FastAPI
      const result = await axios.post("http://localhost:8000/generate-code", { prompt });

      // Log the API response (optional for debugging)
      console.log("API Response:", result.data);

      // Set the response state with the data from the backend
      setResponse(result.data); // Store the full response object
    } catch (error) {
      console.error("Error fetching response:", error);
      setResponse({
        error: true,
        message: "There was an error processing your request."
      });
    } finally {
      setLoading(false); // Clear loading state
    }
  };

  const handleKeyDown = (e) => {
    // Check if Ctrl+Enter is pressed
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault(); // Prevent default behavior (new line)
      if (!loading && prompt.trim()) {
        handleSubmit(e);
      }
    }
  };

  return (
    <div className="chat-container">
      {/* Input form for the prompt */}
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter your prompt here... (Press Ctrl+Enter to submit)"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)} // Update prompt as user types
          onKeyDown={handleKeyDown} // Handle Ctrl+Enter
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Submit"}
        </button>
      </form>

      {/* Display the API response */}
      <div className="response">
        {loading ? (
          <p>Loading responses from models...</p>
        ) : response ? (
          response.error ? (
            <p className="error">{response.message}</p>
          ) : (
            <div className="results-container">
              <h3>Prompt: {response.prompt}</h3>
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Model Name</th>
                    <th>Response</th>
                  </tr>
                </thead>
                <tbody>
                  {response.model_responses && response.model_responses.map((modelResponse, index) => (
                    <tr key={index}>
                      <td className="model-name">{modelResponse.model}</td>
                      <td className="model-response">{modelResponse.response}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : (
          <p>No response yet. Please submit a prompt.</p> // Show default message when no response
        )}
      </div>
    </div>
  );
};

export default ChatBox;
