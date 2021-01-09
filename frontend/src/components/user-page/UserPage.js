import React, { Component } from "react";
import TopBar from "./TopBar"
import ProfilePic from "./ProfilePic"
import PredictionCounter from "./PredictionCounter"
import "./user-page.css"
export default class UserPage extends Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div class="user-page">
                <TopBar />
                <br></br>
                <div class="user-grid">
                    <ProfilePic />
                    <PredictionCounter />
                </div>
                <div class="playlists-grid">
                    <div class="playlist1">
                        PLAYLIST1
                    </div>
                    <div class="playlist2">
                        PLAYLIST2
                    </div>
                    <div class="playlist3">
                        PLAYLIST3
                    </div>
                </div>
                <div class="creds">
                    CREDS
                </div>
            </div>
        )
    }
}