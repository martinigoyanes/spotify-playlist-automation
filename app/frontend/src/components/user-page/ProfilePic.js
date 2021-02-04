import React, { Component } from "react";
import Image from "react-bootstrap/Image";
import "./user-page.css"
export default class ProfilePic extends Component {
    constructor(props) {
        super(props);
    }


    render() {
        return (
            <div class="profile-pic">
                <div>
                    <Image src={this.props.img} roundedCircle />
                </div>
            </div>
        )
    }
}