/****************************************************
* THIS FILE IS PART OF PPL PROJECT 
* cfg.h this file is used to read configuration file 
* 
* Author : linan liuping
*****************************************************/

#include <map>
#include <vector>
#include <string>
#include <list>

/*
这个类用于读取特定格式的配置文件
配置文件是一个普通的文本文件
其大致的格式如下
#这个是注释
#This is comment
[section]
arg1 : val arg2 : val2 arg3 : val2
arg1 : val xxx : val2
[section]
...

基本概念：
section(节）
节是每一组配置在配置文件中的唯一标识，节的名称放在[]之间

parameter-list
每一个节可以拥有多个参数列表，多个参数列表构成参数组。
每一个参数由两部分构成，一是键，一是值，两者使用“：”分开。一个参数列表中的参数同意放在一行上。参数与参数之间用空格隔开
但是键可以由多个单词组成，单词间可以有空格，但多余的空格会被忽略。尚不支持参数值不能包含空格。

parameter-group
多个参数列表，构成一个参数组。

“#” 之后的内容被当做注释，因此请不要在有意义的名称，比如，参数的键或者值中包含“#”

以下是示例代码

//! [0] 构建一个Cfg对象
Cfg cfg("F:/cfg.txt");

//! [1] 判断是否拥有配置信息,即配置文件是否有相关的配置信息
if(!cfg.hasAnyConfig())
{
//! 没有配置信息，此处返回或者使用默认值
}

//! [2] 判断是否有自己感兴趣的信息
if(cfg.hasSection("my_section"))
{
//! [3] 获取配置的数组
auto cfgVec = cfg.sectionConfigVec("my_section");

//! [4] 遍历数组
for(const auto& m : vec)
{
//! 查找感兴趣的配置
auto i = m.find("my_width");

//! 如果找到了则将其转换为自己感兴趣的类型
if (i != m.end())
{
int my_width = std::stoi(i->second);
}

i = m.find("my_height");
if (i != m.end())
{
int m_height = std::stoi(i->second);
}

//! 所有感兴趣的信息都查询完成之后
//! 利用已经获取到的信息，构建参数数据结构，并将其添加到相应的容器中即可
}
}

//! [5] 将构建好的参数数组，传递给gtest框架即可

*/
class Cfg 
{
public:
	typedef std::vector<std::map<std::string, std::string>> cfg_type;
public:
	Cfg()
	{ }

	//! 使用一个配置文件，构造一个Cfg对象
	explicit Cfg(const std::string& filename);

	~Cfg()
	{ }

	//! 让Cfg对象重新加载一个配置文件
	//! 以前的配置信息将会被清空
	void setCfgFile(const std::string& filename);

	//! 判断Cfg对象是否拥有任何配置信息
	//! 如果Cfg对象拥有某些配置信息，则返回值为true否则为false
	bool hasAnyConfig() const { return !m_cfg.empty(); }

	//! 该函数用于检测，Cfg对象是否拥有某个Section的配置信息
	//! 如果有，则返回true否则返回false
	bool hasSection(const std::string& section) const;

	//! 获取Cfg对象所拥有的Section列表
	std::vector<std::string> sectionList() const;

	//! 获取配置数组，如果指定的section不存在，则返回的数组为空。
	const cfg_type& sectionConfigVec(const std::string& section) const;

protected:

public:
	std::map<std::string, cfg_type> m_cfg;
};
