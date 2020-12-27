import React, { Component } from "react";
import { Button } from "@material-ui/core";

export default class SpotifyLogin extends Component {
    constructor(props) {
        super(props);
        this.authenticateSpotify = this.authenticateSpotify.bind(this);
    }

    authenticateSpotify() {
        fetch("/spotify/get-auth-url")
            .then((response) => response.json())
            .then((data) => {
                window.location.replace(data.url);
            });
    }

    render() {
        return <Button onClick={() => this.authenticateSpotify()}>
            Login to Spotify
               </Button>
    }
}