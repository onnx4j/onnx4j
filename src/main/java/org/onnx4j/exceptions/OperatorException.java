/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.exceptions;

import org.onnx4j.exceptions.ErrorCode;
import org.onnx4j.exceptions.ExceptionEnums;
import org.onnx4j.exceptions.Onnx4jException;

public class OperatorException extends Onnx4jException {

	private static final long serialVersionUID = -9110594428291367746L;
	
	private static final String ERR_CODE_PREFIX = "OE";

	private static int ERR_CODE_BASE = 4000;

	public enum NodeExceptionEnums implements ExceptionEnums {

		/**
		 * 警告：注意枚举成员的顺序，不能随意调整！
		 */
		OUTPUT_VALUE_INFO_UNDEFINED("Output of node named %s can not found value info"),
		UNSUPPORTED_ATTRIBUTE_TYPE("Unsupported attribute type named %s");

		public ErrorCode errorCode;
		public String messageTemplate;

		private NodeExceptionEnums(String messageTemplate) {
			this.errorCode = new ErrorCode(ERR_CODE_PREFIX, ERR_CODE_BASE++);
			this.messageTemplate = messageTemplate;
		}

		@Override
		public String getErrorCode() {
			return errorCode.toString();
		}

		@Override
		public String getMessageTemplate() {
			return messageTemplate;
		}

	}

	public OperatorException(NodeExceptionEnums exceptionEnum) {
		this(exceptionEnum, new Object[] {});
	}

	public OperatorException(NodeExceptionEnums exceptionEnum, Object... args) {
		super(exceptionEnum, args);
	}

}