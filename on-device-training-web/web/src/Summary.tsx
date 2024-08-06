import React from "react";

export interface Props {
    text: string;
    summary: string;
    prediction: string;
}

export function Summary(props: Props) {
    const { text, summary, prediction } = props;

    return (
        <div>
            <h5>Text:</h5>
            <p>{text}</p>
            <h5>True Summary:</h5>
            <p>{summary}</p>
            <h5>Predicted Summary:</h5>
            <p>{prediction}</p>
            {prediction === summary ? (
                <span>✅</span>
            ) : (
                <span>❌ Expected {summary}</span>
            )}
        </div>
    );
}
