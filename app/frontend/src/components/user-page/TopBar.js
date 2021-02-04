import React, { Component } from "react";
import Navbar from "react-bootstrap/Navbar";
import Button from "react-bootstrap/Button";
import "./user-page.css"
export default class TopBar extends Component {
    constructor(props) {
        super(props);
        this.logOut = this.logOut.bind(this);
    }

    logOut(){
        fetch("log-out")
            .then((response) => { return response.json(); })
            .then((data) => {
                console.log(data.status);
                window.location.replace('/login');
            });
    }

    render() {
        return (
            <div class="top-bar">
                <Navbar bg="dark" variant="dark">
                    <Navbar.Brand href="#home">AI Playlist Organizer</Navbar.Brand>
                    <Navbar.Collapse className="justify-content-center">
                        <Navbar.Text>
                            Signed in as: <a href="#login">{this.props.username}</a>
                        </Navbar.Text>
                    </Navbar.Collapse>
                    <Button variant="outline-info" onClick={() => this.logOut()}>Log out</Button>
                </Navbar>
            </div>
        )
    }
}