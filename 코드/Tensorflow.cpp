#include <iostream>
#include <string>
#include <nlohmann/json.hpp>
#include <mariadb/conncpp.hpp>
#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <signal.h>
#include <map>

#define BUF_SIZE 1024
#define MAX_CLNT 256
#define Serv_IP "10.10.21.125"
#define PortNumber "34543"

using namespace std;
using nlohmann::json;

int clnt_cnt = 0;
int clnt_socks[MAX_CLNT];
int python_sock;
int client_sock;
int UID = 0;
pthread_mutex_t mutx;


enum EProtocol
    {
        // C# 관련은 10
        CS = 10,
        CS_OpenCV = 11,
        CS_Graph = 12,
        CS_Result = 13,
        DB_Success = 14,
        CS_SaveLog = 15,
        
        // 파이썬 관련 단위 30
        Python_Connect = 21,
        Python_Disconnect = 22,
        Python_Error = 23,
        Python_Result = 24,

        // 오류 관련 단위 30
        JsonWrong_Protocol = 31,
        JsonParsing_Error = 32,
        DB_QueryError = 33,
        DB_UpdateError = 34,
        Image_Error = 35,
        Protocol_Error = 36, 
        DB_Empty = 37,
    };

class DB_CONN
{
private:
    sql::Connection* conn;
public:
    /**
     * DB와의 연결함수이자, 생성자 입니다.
     * @return 없음 */    
    DB_CONN();

    void Save_Log(json obj);
    json Load_Log();
    
    json Fire_info();
};
void Save_Log(string msg);
void Load_Graph();
/**
 * 해당 소켓번호의 클라이언트에게 TCP로 값을 보내는 함수입니다.
 * @param msg json타입으로, 처리가 끝난 값을 클라이언트에게 보냅니다.
 * @param sock 해당 클라이언트의 소켓번호를 기입하면 됩니다.
 * @return 없음
 */
void Send_Client(json msg);
/**
 * 파이썬 클라이언트에게 TCP로 값을 보내는 메소드
 * protocol_check를 통해서 알게 된 파이썬 소켓번호로 값을 보냅니다.
 * @param msg json 타입으로, 클라이언트에게서 받은 msg를 json으로 역직렬화 하여 보냅니다.
 * @param sock 해당 클라이언트의 소켓변수를 기입하여야 합니다.
 * @return 없음
 */
void Send_Python(json msg);
/**
 * 파이썬 클라이언트에게서 TCP로 값을 읽는 메소드
 * protocol_check를 통해서 알게 된 파이썬 소켓번호로부터 값을 읽습니다.
 * @return json 타입으로 전달됩니다.
 */
json Read_Python();
/**
 * 클라이언트 제어용 쓰레드 내부 함수
 * 클라이언트가 접속 종료시 해당 배열에서 클라이언트의 소켓번호를 할당해제한 후
 * 재정렬합니다.
 * @param arg 소켓번호를 전달하기 위해 void타입으로 형변환 후 기입해야합니다.
 * @return 없음
 */
void* handle_clnt(void* arg);
void Clinet_Control();
void Python_Control();


/**
 * 에러 검출시 프로그램 종료하는 코드입니다.
 * @param msg 에러 메시지를 넣으세요 
 * @return 없음 
 */ 
void error_handling(const char* msg);


int main()
{
    signal(SIGPIPE,SIG_IGN);
    int serv_sock, clnt_sock;
    int str_len, i;

    sockaddr_in serv_adr, clnt_adr;
    socklen_t clnt_adr_sz;
    pthread_t t_id;

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family = AF_INET;
    serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_adr.sin_port = htons(atoi(PortNumber));

    if( bind(serv_sock, (sockaddr*)&serv_adr, sizeof(serv_adr)) == -1)
        error_handling("bind() error");
    if( listen(serv_sock,5) == -1)
        error_handling("listen() error");
    printf("Server is Online at Port : %s , IP : %s\n", PortNumber, Serv_IP);
    while(1)
    {
        clnt_adr_sz = sizeof(clnt_adr);
        clnt_sock = accept(serv_sock, (sockaddr*)&clnt_adr, &clnt_adr_sz);
        printf("Connected client IP: %s , Port: %d\n", inet_ntoa(clnt_adr.sin_addr), ntohs(clnt_adr.sin_port));
        pthread_mutex_lock(&mutx);
        clnt_socks[clnt_cnt++] = clnt_sock;
        pthread_mutex_unlock(&mutx);

        pthread_create(&t_id, NULL, handle_clnt, (void*)&clnt_sock);
        pthread_detach(t_id);
    }
    close(serv_sock);
    return 0;
}


DB_CONN::DB_CONN()
{
    try
    {
        sql::Driver* driver = sql::mariadb::get_driver_instance();
        sql::SQLString url("jdbc:mariadb://10.10.21.125:3306/SOLUTION");
        sql::Properties properties({{"user", "FIRE01"}, {"password", "1234"}});
        this->conn = driver->connect(url, properties);
    }
    catch(const sql::SQLException& ex)
    {
        cerr<<ex.what()<<endl;
        cout << "DB connect error" << endl;
        exit(1);
    }
}

void DB_CONN::Save_Log(json obj)
{
    unique_ptr<sql::PreparedStatement>
     stmnt(conn->prepareStatement("INSERT INTO FIRE(FIRE_YES, PERCENT, LOCAL, MOUNTAIN, MONTH) VALUES(?,?,?,?,?);"));
    stmnt->setInt(1, obj["Fire"].get<int>());
    stmnt->setInt(2, obj["result_msg"].get<int>());
    stmnt->setString(3, obj["Local"].get<std::string>());
    stmnt->setString(4, obj["Mountain"].get<std::string>());
    stmnt->setString(5, obj["Month"].get<std::string>());
    try
    {
        stmnt->executeUpdate();
    }
    catch(sql::SQLException ex)
    {
        cerr << "DB_CONN::Save_Log Error : "<< ex.what() << endl;
    }
    
}

json DB_CONN::Load_Log()
{
    json jobject;
    unique_ptr<sql::PreparedStatement>
     stmnt(conn->prepareStatement("SELECT MONTH, MOUNTAIN, LOCAL, FIRE_YES, ACCURACY, LOSS FROM FIRE"));
    try 
    {
        json Month_Array = json::array();
        json Mountain_Array = json::array();
        json Local_Array = json::array();
        // json Fire_Arrray = json::array();
        json Accuracy_Array = json::array();
        json Loss_Array = json::array();
        unique_ptr<sql::ResultSet>res(stmnt -> executeQuery());
        if(res->next())
        {
            do
            {
                Month_Array.push_back(res->getString(1));
                Mountain_Array.push_back(res->getString(2));
                Local_Array.push_back(res->getString(3));
                Accuracy_Array.push_back(res->getFloat(5));
                Loss_Array.push_back(res->getFloat(6));
            } while (res->next());
            jobject["Month"] = Month_Array;
            jobject["Mountain"] = Mountain_Array;
            jobject["Local"] = Local_Array;
            jobject["Accuracy"] = Accuracy_Array;
            jobject["Loss"] = Loss_Array;
            jobject["protocol"] = EProtocol::DB_Success;
            return jobject;
        }
        else
        {
            jobject["protocol"] = EProtocol::DB_Empty;
            return jobject;
        }
    }
    catch(sql::SQLException ex)
    {
        cout << "Load_Log Error : " << ex.what() << endl;
        jobject["protocol"] = EProtocol::DB_QueryError;
        return jobject;
    }
}



/// FIRE로 묶어서 보내는 DB쿼리문이랑 함수 만들어놨어 체그해봐 65번쨰 줄도 체 /////
/// @return 
json DB_CONN::Fire_info()
{
    json firejeck;
    std::vector<json> info;
    std::unique_ptr<sql::PreparedStatement>stmnt(conn->prepareStatement("SELECT MONTH, COUNT(*) AS COUNT FROM FIRE GROUP BY MONTH ORDER BY MONTH"));
    try 
    {
        std::unique_ptr<sql::ResultSet> res5(stmnt->executeQuery());
        while(res5->next())
        {
            json fire_check;
            fire_check["Month"] = res5->getString("MONTH");
            fire_check["count"] = res5->getInt("COUNT");
            info.emplace_back(fire_check);
        }
        firejeck["FIRE"] = info;
        return firejeck;
    }
    catch(sql::SQLException ex)
    {
        cout << "Firo_info Error : " << ex.what() << endl;
        return NULL;
    }
}

void Send_Client(json msg)
{
    string messages = msg.dump();
    send(client_sock, messages.c_str(), messages.length(), 0);
}

void Send_Python(json msg)
{
    msg.erase("protocol");
    string message = msg.dump();
    send(python_sock, message.c_str(), message.length(), 0);
}

json Read_Python()
{
    unique_ptr<char[]> messages (new char[BUF_SIZE]);
    memset(messages.get(), 0, BUF_SIZE);
    recv(python_sock, messages.get(), BUF_SIZE, 0);
    try
    {
        string jsonstirng(messages.get());
        json jobject = json::parse(jsonstirng);
        if(jobject["protocol"] == EProtocol::Python_Error)
        {
            cout << "Read_Python : Protocol_Python Error" << endl;
            return NULL;    
        }
        return jobject;
    }
    catch(const json::exception& e)
    {
        cerr << "Read_Python : Json Parsing error " << e.what() << '\n';
        return NULL;
    }
}

void* handle_clnt(void* arg)
{
    int clnt_sock = *((int*)arg);
    char msg[BUF_SIZE];
    int str_len = read(clnt_sock,msg,BUF_SIZE-1 );
    if( str_len != 0)
    {
        std::cout << str_len << endl;
        if(str_len == -1)
        {
            cerr << "Read Error : " << strerror(errno) << endl;
            return NULL;
        }
        msg[str_len] = 0;
        string serv_msg = msg;
        cout << "handle_clnt : " << serv_msg << endl;
        int temp = stoi(msg);
        switch(temp)
        {
            case EProtocol::CS:
                client_sock = clnt_sock;
                std::cout << "Client_Sock : " << client_sock << endl;
                Clinet_Control();
                break;
            case EProtocol::Python_Connect:
                python_sock = clnt_sock;
                std::cout << "Python_Sock : " << python_sock << endl;
                Python_Control();
                break;
            default:
                std::cout << "Protocol Error : " << temp << endl; 
                break;
        }
        memset(msg, 0, BUF_SIZE); 
    }
    pthread_mutex_lock(&mutx);
    for ( int i = 0; i < clnt_cnt; i++ )
    {
        if(clnt_sock == clnt_socks[i])
        {
            while(i++ < clnt_cnt-1)
                clnt_socks[i] = clnt_socks[i+1];
            break;
        }
    }
    clnt_cnt--;
    pthread_mutex_unlock(&mutx);
    close(clnt_sock);\
    return NULL;
}

void Clinet_Control()
{
    int str_len = 0;
    char msg[BUF_SIZE];
    string serv_msg;
    json obj;
    memset(msg, 0, BUF_SIZE);
    while((str_len = read(client_sock,msg,BUF_SIZE-1)) != 0)
    {
    //    cout << str_len << endl;
        if(str_len == -1)
        {
            cerr << "Read Error : " << strerror(errno) << endl;
        }
        msg[str_len] = 0;
        serv_msg = msg;
        cout << "Clnt_Control : " << serv_msg << endl;
        try
        {
            obj = json::parse(serv_msg);
        }
        catch(json::exception ex)
        {
            cerr << "Client_Control Json Parsing Error : " << ex.what() << endl;
            continue;
        }
        int temp = obj["protocol"];
        cout << temp << endl;
        switch(temp)
        {
            case EProtocol::CS_Graph:
                Load_Graph();
                break;
            case 15:
                Save_Log(serv_msg);
                break;
            default:
                std::cout << "Protocol Error : " << temp << endl; 
                break;
        }
        memset(msg, 0, BUF_SIZE); 
    }
}

void Python_Control()
{
    int str_len = 0;
    char msg[BUF_SIZE];
    string serv_msg;
    memset(msg, 0, BUF_SIZE);
    while((str_len = read(python_sock,msg,BUF_SIZE-1)) != 0)
    {
       cout << str_len << endl;
        if(str_len == -1)
        {
            cerr << "Read Error : " << strerror(errno) << endl;
        }
        msg[str_len] = 0;
        serv_msg = msg;
        cout << "Python_Control : " << serv_msg << endl;
        if(serv_msg != "테스트테스트")
            send(client_sock, msg, BUF_SIZE, 0);
        cout << "클라이언트에 값보냄\n";
        memset(msg, 0, BUF_SIZE); 
    }
    cout << "파이썬 끝남\n";
}

void Load_Graph()
{
    DB_CONN db;
    json jobject = db.Fire_info();
    Send_Client(jobject);
}

void Save_Log(string msg)
{
    DB_CONN db;
    json obj = json::parse(msg);
    db.Save_Log(obj);
}

void error_handling(const char* msg)
{
    cerr << msg << endl;
    exit(1);
}