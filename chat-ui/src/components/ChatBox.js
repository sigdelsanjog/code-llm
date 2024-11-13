import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "./ChatBox.css";

const ChatBox = () => {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  // Function to handle prompt submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Send prompt to FastAPI backend
      const result = await axios.post("http://localhost:8000/generate", { prompt });
      setResponse(result.data.response); // Assuming the backend sends back JSON with a 'response' key
      setPrompt(""); // Clear the prompt input
    } catch (error) {
      console.error("Error fetching response:", error);
      setResponse("There was an error getting a response.");
    }
  };

  return (
    <div className="chat-container">
      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Enter your prompt here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button type="submit">Submit</button>
      </form>
      <div className="response">
        <ReactMarkdown>{response}</ReactMarkdown>
      </div>
    </div>
  );
};

export default ChatBox;
