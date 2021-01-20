import React, { Component } from "react";
import { ScrollView } from "react-native-web"
import TopBar from "./TopBar"
import ProfilePic from "./ProfilePic"
import PredictionCounter from "./PredictionCounter"
import Playlist from "./Playlist"
import "./user-page.css"
export default class UserPage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            name: '',
            img: '',
            predCount: 0
        }
    }

    getProfileInfo() {
        fetch("user-info")
            .then((response) => { return response.json(); })
            .then((data) => { this.setState({ img: data.img,
                                                name: data.name,
                                                predCount: data.predCount 
                                            }); console.log(data); });
    }
    async componentDidMount() {
        this.getProfileInfo();
    } 

    render() {
        return (
            <ScrollView>
                <div class="user-page">
                    <TopBar username={this.state.name}/>
                    <br></br>
                    <div class="user-grid">
                        <ProfilePic img={this.state.img}/>
                        <PredictionCounter predCount={this.state.predCount}/>
                        <p class="creds">
                            Designed by Martín Iglesias.
                        </p>
                    </div>
                </div>
            </ScrollView>
        )
    }
}