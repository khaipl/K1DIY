#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include "game_controller_node.h"

GameControllerNode::GameControllerNode(string name) : rclcpp::Node(name)
{
    _socket = -1;

    declare_parameter<int>("port", 3838);
    declare_parameter<bool>("enable_ip_white_list", false);
    declare_parameter<vector<string>>("ip_white_list", vector<string>{});

    get_parameter("port", _port);
    RCLCPP_INFO(get_logger(), "[get_parameter] port: %d", _port);
    get_parameter("enable_ip_white_list", _enable_ip_white_list);
    RCLCPP_INFO(get_logger(), "[get_parameter] enable_ip_white_list: %d", _enable_ip_white_list);
    get_parameter("ip_white_list", _ip_white_list);
    RCLCPP_INFO(get_logger(), "[get_parameter] ip_white_list(len=%ld)", _ip_white_list.size());
    for (size_t i = 0; i < _ip_white_list.size(); i++)
    {
        RCLCPP_INFO(get_logger(), "[get_parameter]     --[%ld]: %s", i, _ip_white_list[i].c_str());
    }

    _publisher = create_publisher<game_controller_interface::msg::GameControlData>("/booster_soccer/game_controller", 10);
}

GameControllerNode::~GameControllerNode()
{
    if (_socket >= 0)
    {
        close(_socket);
    }

    if (_thread.joinable())
    {
        _thread.join();
    }
}


void GameControllerNode::init()
{
    _socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (_socket < 0)
    {
        RCLCPP_ERROR(get_logger(), "socket failed: %s", strerror(errno));
        throw runtime_error(strerror(errno));
    }

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(_port);

    if (bind(_socket, (sockaddr *)&addr, sizeof(addr)) < 0)
    {
        RCLCPP_ERROR(get_logger(), "bind failed: %s (port=%d)", strerror(errno), _port);
        throw runtime_error(strerror(errno));
    }

    RCLCPP_INFO(get_logger(), "Listening for UDP broadcast on 0.0.0.0:%d", _port);
    RCLCPP_INFO(get_logger(), "Expected RoboCupGameControlData size: %ld", sizeof(RoboCupGameControlData));

    _thread = thread(&GameControllerNode::spin, this);
}

void GameControllerNode::spin()
{
    sockaddr_in remote_addr;
    socklen_t remote_addr_len = sizeof(remote_addr);

    RoboCupGameControlData data;
    game_controller_interface::msg::GameControlData msg;

    while (rclcpp::ok())
    {
        ssize_t ret = recvfrom(_socket, &data, sizeof(data), 0, (sockaddr *)&remote_addr, &remote_addr_len);
        if (ret < 0)
        {
            RCLCPP_ERROR(get_logger(), "receiving UDP message failed: %s", strerror(errno));
            continue;
        }

        string remote_ip = inet_ntoa(remote_addr.sin_addr);

        if (ret != sizeof(data))
        {
            RCLCPP_INFO(get_logger(), "packet from %s invalid length=%ld, expected=%ld", remote_ip.c_str(), ret, sizeof(data));
            continue;
        }

        if (data.version != GAMECONTROLLER_STRUCT_VERSION)
        {
            RCLCPP_INFO(get_logger(), "packet from %s invalid version: %d", remote_ip.c_str(), data.version);
            continue;
        }

        if (!check_ip_white_list(remote_ip))
        {
            RCLCPP_INFO(get_logger(), "received packet from %s, but not in ip white list, ignore it", remote_ip.c_str());
            continue;
        }

        handle_packet(data, msg);

        _publisher->publish(msg);

        RCLCPP_INFO(get_logger(), "handle packet successfully ip=%s, packet_number=%d", remote_ip.c_str(), data.packetNumber);
    }
}


bool GameControllerNode::check_ip_white_list(string ip)
{
    if (!_enable_ip_white_list)
    {
        return true;
    }
    for (size_t i = 0; i < _ip_white_list.size(); i++)
    {
        if (ip == _ip_white_list[i])
        {
            return true;
        }
    }
    return false;
}

void GameControllerNode::handle_packet(RoboCupGameControlData &data, game_controller_interface::msg::GameControlData &msg)
{

    // header is fixed length 4
    for (int i = 0; i < 4; i++)
    {
        msg.header[i] = data.header[i];
    }
    msg.version = data.version;
    msg.packet_number = data.packetNumber;
    msg.players_per_team = data.playersPerTeam;
    msg.competition_phase = data.competitionPhase;
    msg.competition_type = data.competitionType;
    msg.game_phase = data.gamePhase;
    msg.state = data.state;
    msg.set_play = data.setPlay;
    msg.first_half = data.firstHalf;
    msg.kicking_team = data.kickingTeam;
    
    msg.secs_remaining = data.secsRemaining;
    msg.secondary_time = data.secondaryTime;

    // teams is fixed length 2
    for (int i = 0; i < 2; i++)
    {
        msg.teams[i].team_number = data.teams[i].teamNumber;
        msg.teams[i].field_player_colour = data.teams[i].fieldPlayerColour;
        msg.teams[i].goalkeeper_colour = data.teams[i].goalkeeperColour;
        msg.teams[i].goalkeeper = data.teams[i].goalkeeper;
        msg.teams[i].score = data.teams[i].score;
        msg.teams[i].penalty_shot = data.teams[i].penaltyShot;
        msg.teams[i].single_shots = data.teams[i].singleShots;
        

        msg.teams[i].message_budget = data.teams[i].messageBudget;

        int players_len = sizeof(data.teams[i].players) / sizeof(data.teams[i].players[0]);
        msg.teams[i].players.clear(); // because msg is reused, remember to clear it
        for (int j = 0; j < players_len; j++)
        {
            game_controller_interface::msg::RobotInfo rf;
            rf.penalty = data.teams[i].players[j].penalty;
            rf.secs_till_unpenalised = data.teams[i].players[j].secsTillUnpenalised;
            msg.teams[i].players.push_back(rf);
        }
    }
}