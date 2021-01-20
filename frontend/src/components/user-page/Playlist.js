import React, { Component } from "react";
import "./playlist.css"
export default class Playlist extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div class="playlist-holder">
                <div>
                    <h1> Playlist_name</h1>
                </div>
                <div>
                    Song1
                </div>
            </div>
        )
    }
}