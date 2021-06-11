all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

check_src:
	@Makefile.scripts/check_src.sh

install_anaconda3:
	@Makefile.scripts/install_anaconda3.sh

install_ompl:
	@Makefile.scripts/install_ompl.sh

install_yarr:
	@Makefile.scripts/install_yarr.sh

install: check_src install_anaconda3 install_ompl install_yarr
	@Makefile.scripts/install.sh
