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
��������ڶ�ȡ�ض���ʽ�������ļ�
�����ļ���һ����ͨ���ı��ļ�
����µĸ�ʽ����
#�����ע��
#This is comment
[section]
arg1 : val arg2 : val2 arg3 : val2
arg1 : val xxx : val2
[section]
...

�������
section(�ڣ�
����ÿһ�������������ļ��е�Ψһ��ʶ���ڵ����Ʒ���[]֮��

parameter-list
ÿһ���ڿ���ӵ�ж�������б���������б��ɲ����顣
ÿһ�������������ֹ��ɣ�һ�Ǽ���һ��ֵ������ʹ�á������ֿ���һ�������б��еĲ���ͬ�����һ���ϡ����������֮���ÿո����
���Ǽ������ɶ��������ɣ����ʼ�����пո񣬵�����Ŀո�ᱻ���ԡ��в�֧�ֲ���ֵ���ܰ����ո�

parameter-group
��������б�����һ�������顣

��#�� ֮������ݱ�����ע�ͣ�����벻Ҫ������������ƣ����磬�����ļ�����ֵ�а�����#��

������ʾ������

//! [0] ����һ��Cfg����
Cfg cfg("F:/cfg.txt");

//! [1] �ж��Ƿ�ӵ��������Ϣ,�������ļ��Ƿ�����ص�������Ϣ
if(!cfg.hasAnyConfig())
{
//! û��������Ϣ���˴����ػ���ʹ��Ĭ��ֵ
}

//! [2] �ж��Ƿ����Լ�����Ȥ����Ϣ
if(cfg.hasSection("my_section"))
{
//! [3] ��ȡ���õ�����
auto cfgVec = cfg.sectionConfigVec("my_section");

//! [4] ��������
for(const auto& m : vec)
{
//! ���Ҹ���Ȥ������
auto i = m.find("my_width");

//! ����ҵ�������ת��Ϊ�Լ�����Ȥ������
if (i != m.end())
{
int my_width = std::stoi(i->second);
}

i = m.find("my_height");
if (i != m.end())
{
int m_height = std::stoi(i->second);
}

//! ���и���Ȥ����Ϣ����ѯ���֮��
//! �����Ѿ���ȡ������Ϣ�������������ݽṹ����������ӵ���Ӧ�������м���
}
}

//! [5] �������õĲ������飬���ݸ�gtest��ܼ���

*/
class Cfg 
{
public:
	typedef std::vector<std::map<std::string, std::string>> cfg_type;
public:
	Cfg()
	{ }

	//! ʹ��һ�������ļ�������һ��Cfg����
	explicit Cfg(const std::string& filename);

	~Cfg()
	{ }

	//! ��Cfg�������¼���һ�������ļ�
	//! ��ǰ��������Ϣ���ᱻ���
	void setCfgFile(const std::string& filename);

	//! �ж�Cfg�����Ƿ�ӵ���κ�������Ϣ
	//! ���Cfg����ӵ��ĳЩ������Ϣ���򷵻�ֵΪtrue����Ϊfalse
	bool hasAnyConfig() const { return !m_cfg.empty(); }

	//! �ú������ڼ�⣬Cfg�����Ƿ�ӵ��ĳ��Section��������Ϣ
	//! ����У��򷵻�true���򷵻�false
	bool hasSection(const std::string& section) const;

	//! ��ȡCfg������ӵ�е�Section�б�
	std::vector<std::string> sectionList() const;

	//! ��ȡ�������飬���ָ����section�����ڣ��򷵻ص�����Ϊ�ա�
	const cfg_type& sectionConfigVec(const std::string& section) const;

protected:

public:
	std::map<std::string, cfg_type> m_cfg;
};
