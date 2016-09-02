#include "cfg.h"

#include <cctype>
#include <cassert>
#include <iostream>
#include <fstream>
namespace {
	enum TxtLineType {
		Unknown ,
		Empty ,
		Comment ,
		Section , 
		Arguments
	};

	TxtLineType PredictTxtLineType(const std::string& txtline)
	{
		if (txtline.empty())
		{
			return Empty;
		}

		for (std::size_t i = 0; i < txtline.size(); ++i)
		{
           
			if (!std::isspace(txtline[i]))
			{
				if (txtline[i] == '#')
				{
					return Comment;
				}
				else if (txtline[i] == '[')
				{
					return Section;
				}
				else if (std::isalpha(txtline[i]))
				{
         
					return Arguments;
				}
				else
				{
					return Unknown;
				}
			}
		}
		return Empty;
	}

	std::map<std::string, std::string> parseArgsLine(const std::string& txtline)
	{
		std::map<std::string, std::string> ret;
		std::string key;
		std::string val;
		bool bParseKey = true;
		for (std::size_t i = 0; i < txtline.size(); ++i)
		{
			if (!std::isspace(txtline[i]))
			{
				if (txtline[i] == '#')
				{
					if (!key.empty())
					{
						if (key.back() == ' ')
						{
							key.pop_back();
						}
						if (!key.empty())
						{
							ret.emplace(key, val);
						}
					}
					return ret;
				}
				else if (txtline[i] == ':')
				{
					bParseKey = false;
				}
				else
				{
					if (bParseKey)
					{
						key.push_back(txtline[i]);
					}
					else
					{
						val.push_back(txtline[i]);
					}
				}
			}
			else
			{
				if (key.empty())
				{
					continue;
				}

				if (bParseKey)
				{
					if (key.back() != ' ')
					{
						key.push_back(' ');
					}
				}
				else 
				{
					if (!key.empty())
					{
						if (key.back() == ' ')
						{
							key.pop_back();
						}
					}
					if (!key.empty())
					{
						ret.emplace(key, val);
						key.clear();
					}
					val.clear();
					bParseKey = true;
				}
			}
		}
		if (!key.empty())
		{
			if (key.back() == ' ')
			{
				key.pop_back();
			}
		}
		if (!key.empty())
		{
			ret.emplace(key, val);
		}
		return ret;
	}

	std::string parseSecLine(const std::string& txtline)
	{
		std::size_t _idx = txtline.find_first_of('[');
		assert(_idx != std::string::npos);
		for (++_idx;_idx < txtline.size();++_idx)
		{
			if (isprint(txtline[_idx]))
			{
				break;
			}
		}

		std::string ret;
		for (; _idx < txtline.size(); ++_idx)
		{
			if (txtline[_idx] == ']')
			{
				break;
			}
			else if (txtline[_idx] == '#')
			{
#ifdef DEBUG
				std::cerr << "Error!!!!" << std::endl;
#endif
				break;
			}
			else
			{
				if (std::isspace(txtline[_idx]))
				{
					if (!ret.empty() && ret.back() != ' ')
					{
						ret.push_back(' ');
					}
				}
				else
				{
					ret.push_back(txtline[_idx]);
				}
			}
		}

		if (ret.back() == ' ')
		{
			ret.pop_back();
		}
		return ret;
	}
}

Cfg::Cfg(const std::string& filename)
{
	setCfgFile(filename);
}

void Cfg::setCfgFile(const std::string& filename)
{
	std::fstream inf(filename, std::ios::in);
    if(!inf)
        std::cout<<"file open failed!"<<std::endl;
    //std::cout<<"dddddd"<<std::endl;
	std::string line;
	std::string sec;
	cfg_type args;
    
    m_cfg.clear();
    //if(!m_cfg.empty())
	  //  m_cfg.clear();
   // std::cout<<"while start."<<std::endl;
	while(std::getline(inf, line))
	{
        
		auto linetype = PredictTxtLineType(line);

        std::cout<<linetype<<std::endl;
		if (linetype == Section)
		{
            std::cout<<"section:"<<line;
			if (!sec.empty() && !args.empty())
			{
				m_cfg.emplace(sec, std::move(args));
			}
			sec = parseSecLine(line);
		}
		else if (linetype == Arguments)
		{
            //printf("argumet %s\n",line);
            std::cout<<"argument:"<<line<<std::endl;
			if (!sec.empty())
			{
				auto args_map = parseArgsLine(line);
				if (!args_map.empty())
				{
					args.emplace_back(std::move(args_map));
				}
			}
		}
	}
	if (!sec.empty() && !args.empty())
	{
		m_cfg.emplace(sec, std::move(args));
	}
    std::cout<<"read txt complete!"<<std::endl;
}

bool Cfg::hasSection(const std::string& section) const
{
	auto i = m_cfg.find(section);
	return i != m_cfg.end();
}

std::vector<std::string> Cfg::sectionList() const
{
	std::vector<std::string> ret;
	for (auto i = m_cfg.begin(); i != m_cfg.end(); ++i)
	{
		ret.emplace_back(i->first);
	}
	return ret;
}

const Cfg::cfg_type& Cfg::sectionConfigVec(const std::string& section) const
{
	auto i = m_cfg.find(section);
	if (i == m_cfg.end())
	{
		return cfg_type();
	}
	else
	{
		return i->second;
	}
}

#ifdef TEST
#endif

/*
int main()
{
	const char* str1 = "## This is a comment line!";
	const char* str2 = "[ test_name ]";
	const char* str3 = " arg1:2 arg3:4 arg4:5 cctt:789#23423432";
	const char* str4 = "";
	{
		assert(Comment == PredictTxtLineType(str1));
		assert(Section == PredictTxtLineType(str2));
		assert(Arguments == PredictTxtLineType(str3));
		assert(Empty == PredictTxtLineType(str4));
	}


	{
		assert(std::string("test_name") == parseSecLine(str2));
	}

	
    auto k = parseArgsLine(str3);
	
	Cfg cfg("F:/cfg.txt");

	for (const std::string& i : cfg.sectionList())
	{
		std::cout << i << std::endl;
	}

	if (cfg.hasSection("test_name1"))
	{
		auto vec = cfg.sectionConfigVec("test_name1");
		std::cout << "size" << vec.size() << std::endl;
		if (!vec.empty())
		{
			for (const auto& m : vec)
			{
				auto i = m.find("param!");
				if (i != m.end())
				{
					//! 将second字符串解析成，自己想要的数据
					//i->second;
				}

			}
		}
	}

	return 0;
}*/
