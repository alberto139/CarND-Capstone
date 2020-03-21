// Generated by gencpp from file dbw_mkz_msgs/TurnSignalCmd.msg
// DO NOT EDIT!


#ifndef DBW_MKZ_MSGS_MESSAGE_TURNSIGNALCMD_H
#define DBW_MKZ_MSGS_MESSAGE_TURNSIGNALCMD_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <dbw_mkz_msgs/TurnSignal.h>

namespace dbw_mkz_msgs
{
template <class ContainerAllocator>
struct TurnSignalCmd_
{
  typedef TurnSignalCmd_<ContainerAllocator> Type;

  TurnSignalCmd_()
    : cmd()  {
    }
  TurnSignalCmd_(const ContainerAllocator& _alloc)
    : cmd(_alloc)  {
  (void)_alloc;
    }



   typedef  ::dbw_mkz_msgs::TurnSignal_<ContainerAllocator>  _cmd_type;
  _cmd_type cmd;





  typedef boost::shared_ptr< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> const> ConstPtr;

}; // struct TurnSignalCmd_

typedef ::dbw_mkz_msgs::TurnSignalCmd_<std::allocator<void> > TurnSignalCmd;

typedef boost::shared_ptr< ::dbw_mkz_msgs::TurnSignalCmd > TurnSignalCmdPtr;
typedef boost::shared_ptr< ::dbw_mkz_msgs::TurnSignalCmd const> TurnSignalCmdConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator1> & lhs, const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator2> & rhs)
{
  return lhs.cmd == rhs.cmd;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator1> & lhs, const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dbw_mkz_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "f1310dcd252c98fc408c6df907b9495a";
  }

  static const char* value(const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xf1310dcd252c98fcULL;
  static const uint64_t static_value2 = 0x408c6df907b9495aULL;
};

template<class ContainerAllocator>
struct DataType< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dbw_mkz_msgs/TurnSignalCmd";
  }

  static const char* value(const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Turn signal command enumeration\n"
"TurnSignal cmd\n"
"\n"
"================================================================================\n"
"MSG: dbw_mkz_msgs/TurnSignal\n"
"uint8 value\n"
"\n"
"uint8 NONE=0\n"
"uint8 LEFT=1\n"
"uint8 RIGHT=2\n"
;
  }

  static const char* value(const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.cmd);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TurnSignalCmd_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dbw_mkz_msgs::TurnSignalCmd_<ContainerAllocator>& v)
  {
    s << indent << "cmd: ";
    s << std::endl;
    Printer< ::dbw_mkz_msgs::TurnSignal_<ContainerAllocator> >::stream(s, indent + "  ", v.cmd);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DBW_MKZ_MSGS_MESSAGE_TURNSIGNALCMD_H
