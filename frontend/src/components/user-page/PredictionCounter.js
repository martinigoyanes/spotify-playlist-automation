import React, { Component } from "react";
import "./user-page.css"
export default class PredictionCounter extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div class="pred-count">
                <h1>
                    Currently the AI Model has done: <br></br>
                    None Predictions
                </h1>
            </div>
        )
    }
}