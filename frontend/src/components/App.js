import React, { Component } from "react";
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link,
    Redirect,
} from "react-router-dom";
import { render } from "react-dom";
import HomePage from "./home-page/HomePage"
import UserPage from "./user-page/UserPage"
import "bootstrap/dist/css/bootstrap.min.css";



export default class App extends Component{
    constructor(props) {
        super(props);
        this.state = {
            spotifyAuthd: false,
        };
    }
    authenticated() {
        fetch("is-authenticated")
            .then((response) => { return response.json(); })
            .then((data) => { this.setState({ spotifyAuthd: data.status }); console.log(data); });
    }
    async componentDidMount() {
        this.authenticated();
    } 

    render(){
        return(
         <Router>
            <Switch>
                <Route exact path="/" render={() => {
                    return this.state.spotifyAuthd ? (
                            <Redirect to="/user-page" />
                        ) : (
                            <Redirect to="/login" />
                        );
                }}  />
                <Route path="/login" render={() => {
                    return this.state.spotifyAuthd ? (
                            <Redirect to="/user-page" />
                        ) : (
                            <HomePage />
                        );
                }}  />
                <Route path="/user-page" render={() => {
                    return this.state.spotifyAuthd ? (
                            <UserPage />
                        ) : (
                            <Redirect to="/login" />
                        );
                }}  />
            </Switch>
        </Router>
        )
    }
}
const appDiv = document.getElementById("app");
render(<App />, appDiv);