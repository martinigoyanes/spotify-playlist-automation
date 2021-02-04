import React, { Component } from "react";
import ReactTypingEffect from 'react-typing-effect';
import SpotifyLogin from "./SpotifyLogin"
import "./home-page.css"

export default class HomePage extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div class="home-page">
                <div class="app-name-home">
                    <h1>
                        <ReactTypingEffect
                            text={"AI"}
                            typingDelay={100}
                            eraseDelay={8000}
                            typingSpeed={50}
                            cursor={' '}
                        />
                        <br></br>
                        <ReactTypingEffect
                            text={"Playlist"}
                            typingDelay={200}
                            eraseDelay={7000}
                            typingSpeed={50}
                            eraseSpeed={25}
                            cursor={' '}
                        />
                        <br></br>
                        <ReactTypingEffect
                            text={"Organizer"}
                            typingDelay={400}
                            eraseDelay={5000}
                            typingSpeed={50}
                            eraseSpeed={25}
                            cursor={' '}
                        />
                        <br></br>
                    </h1>
                </div>
                {/* <div class="app-description">
                    <h1>
                    Uses a Deep Neural Network trained on your music taste.
                    <br></br>
                    Tracks your Spotify usage and automatically predicts the correct playlist for each song.
                    </h1>
                </div> */}
                <SpotifyLogin />
                <p class="description">
                    Designed by Martín Iglesias.
                </p>
            </div>
        )
    }
}