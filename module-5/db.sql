/*
SQLyog Community Edition- MySQL GUI v7.01 
MySQL - 5.0.27-community-nt : Database - cctvsurvillience
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`cctvsurvillience` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `cctvsurvillience`;

/*Table structure for table `readingcount` */

DROP TABLE IF EXISTS `readingcount`;

CREATE TABLE `readingcount` (
  `uid` int(11) default NULL,
  `ht_count` int(11) default '0',
  `toi_count` int(11) default '0',
  `ie_count` int(11) default '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `readingcount` */

/*Table structure for table `userdetails` */

DROP TABLE IF EXISTS `userdetails`;

CREATE TABLE `userdetails` (
  `uid` int(11) NOT NULL auto_increment,
  `name` varchar(100) default NULL,
  `address` varchar(100) default NULL,
  `email` varchar(100) default NULL,
  `mobile` varchar(100) default NULL,
  `password` varchar(100) default NULL,
  `ht_count` int(11) default '0',
  `toi_count` int(11) default '0',
  `ie_count` int(11) default '0',
  PRIMARY KEY  (`uid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `userdetails` */

insert  into `userdetails`(`uid`,`name`,`address`,`email`,`mobile`,`password`,`ht_count`,`toi_count`,`ie_count`) values (1,'abcd','mumbai','abc@gmail.com','7894561230','123',8,8,6);

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
