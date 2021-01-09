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
                <Image src="https://platform-lookaside.fbsbx.com/platform/profilepic/?asid=1759772620905327&height=300&width=300&ext=1612734033&hash=AeRUEGmE4OMfq5BaLRE" roundedCircle />
            </div>
        )
    }
}