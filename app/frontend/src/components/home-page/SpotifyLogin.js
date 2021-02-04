import React, { Component } from "react";
import "./home-page.css"
export default class SpotifyLogin extends Component {
    constructor(props) {
        super(props);
        this.authenticateSpotify = this.authenticateSpotify.bind(this);
    }

    authenticateSpotify() {
        fetch("get-auth-url")
            .then((response) => response.json())
            .then((data) => {
                window.location.replace(data.url);
            });
    }

    render() {
        return <div class="login">
                <button  class="SpotifyButton" onClick={() => this.authenticateSpotify()}>
                Login with Spotify
                </button>
            </div>
    }
}