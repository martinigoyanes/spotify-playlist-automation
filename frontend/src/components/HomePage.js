import React, { Component } from "react";
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link,
    Redirect,
} from "react-router-dom";
import SpotifyLogin from  "./SpotifyLogin"
import UserPage from "./UserPage"

export default class HomePage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            spotifyAuthd: false,
        };
    }

    async componentDidMount(){
        fetch("/spotify/is-authenticated")
        .then((response) => { return response.json(); })
        .then((data) => { this.setState({spotifyAuthd: data.status}); console.log(this.spotifyAuthd);})
    }

    render() {
        return (
            <Router>
                <Switch>
                    <Route exact path ="/" render={() =>  { return this.state.spotifyAuthd ? (
                        <Redirect to="/user-page" /> ) : (
                            <SpotifyLogin/>
                        );
                }} />
                    <Route path="/user-page" component={UserPage}/>
                </Switch>
            </Router>
        )
    }
}